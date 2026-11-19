# -*- coding: utf-8 -*-
"""
모바일 엔진 (PC 21.13 의 번호 선정 로직을 그대로 옮긴 것: 엑셀/화면 코드 제외)
브라우저(Pyodide) 안에서 실행되며 app.js 가 아래 함수를 호출한다. (build_mobile_engine.py 로 만든 파일: 직접 고치지 않는다)
"""
import itertools
import json
from functools import lru_cache
from typing import List, Dict, Any, Optional, Callable, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_SEED_BASE = 700000                # 무작위 기준선 시드 = RANDOM_SEED_BASE + 학습 회차 수 (모델별 세트와 같은 시드 규칙)

FloatArray = NDArray[np.float32]

P_BASE = 6.0 / 45.0

MODEL_KEYS = ["gb", "knn", "nb", "rf", "dt"]   # 모델 5개 (같은 15개 특징, 알고리즘만 다름)

# 5세트: (모델, 이름, 설명) — 순서는 최신 100회 평균 일치 순위
SET_MODELS: List[Tuple[str, str, str]] = [
    ("gb", "그래디언트 부스팅", "트리를 차례로 보강하는 모델"),
    ("knn", "k-최근접 이웃", "비슷한 과거 300개의 평균"),
    ("nb", "나이브 베이즈", "특징을 독립으로 보는 확률 모델"),
    ("rf", "랜덤 포레스트", "여러 트리의 투표"),
    ("dt", "결정 트리(얕음)", "규칙 몇 개로 나누는 얕은 트리"),
]

AI_KEY = "hybrid"

PREV_CAP = 6                             # 한 세트에 넣을 수 있는 직전 회차 번호의 최대 개수 (6 = 상한 없음)
POOL_WINDOW = 52                         # 풀을 만드는 최근 회차 수 (약 1년)
POOL_SIZE = 36                           # 풀에 넣는 번호 수 (최근 POOL_WINDOW회에서 가장 많이 나온 순)

# 학습 시간 비중(실측): 진행률을 단계 수가 아니라 실제 걸리는 시간에 맞춰 보고하기 위한 값
FIT_WEIGHTS = {"gb": 0.70, "knn": 0.05, "nb": 0.02, "rf": 0.20, "dt": 0.03}


def pick_top(p: NDArray[np.floating[Any]], k: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """확률 상위 k개 번호의 인덱스. 확률이 같으면 무작위 순서로 가른다(작은 번호 쏠림 방지)."""
    order = rng.permutation(len(p))
    return order[np.argsort(-p[order], kind="stable")][:k]


def pick_top_capped(p: NDArray[np.floating[Any]], k: int, rng: np.random.Generator,
                    prev: Set[int], cap: int) -> NDArray[np.intp]:
    """pick_top 과 같지만 직전 회차 번호(prev)는 cap개까지만 넣는다. 넘는 번호는 건너뛰고 다음 순위 번호로 채운다.
    직전 번호가 cap개 이하이면 pick_top 과 결과가 같다."""
    out: List[int] = []
    count = 0
    for i in pick_top(p, len(p), rng):
        if int(i) + 1 in prev:
            if count >= cap:
                continue
            count += 1
        out.append(int(i))
        if len(out) == k:
            break
    return np.asarray(out, dtype=np.intp)


@lru_cache(maxsize=8)
def _combo_index(k: int) -> NDArray[np.int8]:
    """k개 후보 중 6개를 고르는 모든 조합의 위치 (k=14: 3,003개, 22: 74,613개, 26: 230,230개)"""
    return np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(k), 6)), dtype=np.int8).reshape(-1, 6)


def shape_ok(x: NDArray[np.int16]) -> NDArray[np.bool_]:
    """x: (N,6) 오름차순 번호. 구조 조건을 모두 통과하는 조합만 True"""
    d = np.diff(x, axis=1)
    run = np.ones(len(x), np.int16)
    cur = np.ones(len(x), np.int16)
    for j in range(5):
        cur = np.where(d[:, j] == 1, cur + 1, 1)
        run = np.maximum(run, cur)
    odd = (x % 2).sum(1)
    total = x.sum(1)
    band = np.floor_divide(x - 1, 10)
    groups = np.stack([(band == k).sum(1) for k in range(5)], axis=1)        # 1~10 / 11~20 / 21~30 / 31~40 / 41~45
    return ((run < 3) & ((d == 1).sum(1) < 3) & (odd > 0) & (odd < 6) & (total >= 80) & (total <= 200)
            & (groups.max(1) < 4) & ((groups > 0).sum(1) > 2))


def pick_best(score: NDArray[np.floating[Any]], rng: np.random.Generator, prev: Set[int], allowed: NDArray[np.bool_]) -> List[int]:
    """allowed(풀) 안의 번호로 만든 6개 조합 중 구조 조건과 직전 번호 상한(PREV_CAP)을 모두 통과하고 점수 합이 가장 큰 조합 (번호 1~45).
    점수가 같으면 rng 순서로 가른다. 상위 후보를 넓혀 가며 찾고, 후보 밖 번호를 쓰는 조합이 더 높을 수 없음을 확인하면 멈춘다."""
    nums = np.flatnonzero(allowed) + 1
    nums = nums[rng.permutation(len(nums))]
    order = nums[np.argsort(-score[nums - 1], kind="stable")]
    prev_mask = np.zeros(46, dtype=bool)
    prev_mask[list(prev)] = True
    found: List[int] = []
    for k in (14, 18, 22, 26, len(order)):
        k = min(k, len(order))
        cand = order[:k]
        x = np.sort(cand[_combo_index(k)].astype(np.int16), axis=1)
        ok = shape_ok(x) & (prev_mask[x].sum(1) <= PREV_CAP)
        if ok.any():
            total = np.where(ok, score[x - 1].sum(1), -np.inf)
            best = int(np.argmax(total))
            found = [int(n) for n in x[best]]
            bound = float(score[cand[:5] - 1].sum() + score[order[k] - 1]) if k < len(order) else -np.inf
            if total[best] >= bound - 1e-12:
                return found
        if k == len(order):
            break
    if found:
        return found
    return sorted((pick_top_capped(np.where(allowed, score, score.min() - 1.0), 6, rng, prev, PREV_CAP) + 1).tolist())   # 조건을 모두 지키는 조합이 없을 때만


def random_sets(t: int) -> List[List[int]]:
    """무작위 기준선: 학습 회차 수 t 에 대해 항상 같은 무작위 5세트 (직전 번호 제한 없는 순수 무작위)"""
    rng = np.random.default_rng(RANDOM_SEED_BASE + t)
    return [sorted(int(n) + 1 for n in rng.choice(45, size=6, replace=False)) for _ in range(5)]


def match_counts(sets: List[List[int]], actual: List[int]) -> List[int]:
    win = set(actual)
    return [len(win & set(s)) for s in sets]



class LottoEngine:
    WINDOWS = (1, 3, 5, 10, 20, 50, 100)
    MIN_T = 50                      # 특징 계산을 위한 워밍업 구간 (이 이전 회차는 이력으로만 사용)

    def __init__(self):
        self.past_data: List[List[int]] = []
        self.bonus_data: List[int] = []
        self.rounds: List[int] = []
        self.last_info: Dict[str, Any] = {}
        self.m: Optional[NDArray[np.int8]] = None      # (T,45) 당첨 여부
        self.b: Optional[NDArray[np.int8]] = None      # (T,45) 보너스 여부
        self.x: Optional[FloatArray] = None             # (T+1,45,F) 특징 (t회차 이전 정보만 사용)
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------ 데이터
    def load_rows(self, rows: List[List[int]]) -> None:
        """rows: [회차, 번호1..번호6, 보너스] 목록 (엑셀과 같은 열 구조)"""
        seen = {}
        for r in rows:
            seen[int(r[0])] = [int(v) for v in r[:8]]
        if not seen:
            raise ValueError("유효한 회차 데이터가 없습니다.")
        past, bonuses, rounds = [], [], []
        for rnd in sorted(seen):
            _, *nums, bonus = seen[rnd]
            if len(set(nums)) != 6:
                raise ValueError(f"{rnd}회: 번호 중복")
            if any(not (1 <= n <= 45) for n in nums + [bonus]):
                raise ValueError(f"{rnd}회: 범위 밖 번호")
            if bonus in nums:
                raise ValueError(f"{rnd}회: 보너스가 본번호와 중복")
            past.append(sorted(nums))
            bonuses.append(bonus)
            rounds.append(rnd)
        self.past_data, self.bonus_data, self.rounds = past, bonuses, rounds
        self.last_info = {"round": rounds[-1], "nums": past[-1], "bonus": bonuses[-1]}
        self._prepare()

    def _prepare(self) -> None:
        T = len(self.past_data)
        m_arr = np.zeros((T, 45), dtype=np.int8)
        b_arr = np.zeros((T, 45), dtype=np.int8)
        for i, (row, bn) in enumerate(zip(self.past_data, self.bonus_data)):
            row_idx = np.asarray(row, dtype=np.int64) - 1
            m_arr[i, row_idx] = 1
            b_arr[i, bn - 1] = 1
        self.m, self.b = m_arr, b_arr
        self._build_features()

    # ------------------------------------------------------------------ 특징
    def _build_features(self) -> None:
        """
        X[t, n, :] = 't번째 추첨 직전까지의 정보만으로' 계산한 번호 n의 특징.
        (누적합 기반이라 미래 정보가 섞일 수 없다. t = T 는 아직 안 나온 다음 회차)
        """
        if self.m is None or self.b is None:
            raise RuntimeError("당첨 데이터가 준비되지 않았습니다.")

        m_float = self.m.astype(np.float32)
        b_float = self.b.astype(np.float32)
        T = m_float.shape[0]
        idx = np.arange(T + 1)

        cum = np.zeros((T + 1, 45), dtype=np.float32)
        cum[1:] = np.cumsum(m_float, axis=0)
        bonus_cum = np.zeros((T + 1, 45), dtype=np.float32)
        bonus_cum[1:] = np.cumsum(b_float, axis=0)

        def rate(cum_arr: FloatArray, w: int) -> FloatArray:
            lo = np.maximum(idx - w, 0)
            cnt = cum_arr[idx] - cum_arr[lo]
            den = np.maximum(np.minimum(idx, w), 1)[:, None]
            return cnt / den

        feats, names = [], []
        for w in self.WINDOWS:
            feats.append(rate(cum, w)); names.append(f"출현율_{w}회")

        rate_all = cum / np.maximum(idx, 1)[:, None]
        feats.append(rate_all); names.append("출현율_전체")

        gap = np.empty((T + 1, 45), dtype=np.float32)       # 마지막 출현 이후 경과 회차 (직전 회차면 1)
        last = np.full(45, -1.0, dtype=np.float32)
        for t in range(T + 1):
            gap[t] = t - last
            if t < T:
                last[m_float[t] > 0] = t
        feats.append(np.minimum(gap, 60.0)); names.append("미출현_간격")
        feats.append(np.minimum(gap * rate_all, 8.0)); names.append("간격/평균간격")

        z_all = (cum - idx[:, None] * P_BASE) / np.sqrt(np.maximum(idx, 1)[:, None] * P_BASE * (1 - P_BASE))
        feats.append(z_all); names.append("출현편차_z")

        prev = np.zeros((T + 1, 45), dtype=np.float32)
        prev[1:] = m_float
        nb = np.zeros_like(prev)
        nb[:, 1:] += prev[:, :-1]
        nb[:, :-1] += prev[:, 1:]
        feats.append(nb); names.append("직전_인접수")

        grp = np.floor_divide(np.arange(45), 10)
        group_matrix = np.zeros((45, 5), dtype=np.float32)
        group_matrix[np.arange(45), grp] = 1.0
        lo20 = np.maximum(idx - 20, 0)
        cnt20 = cum[idx] - cum[lo20]
        gsum = cnt20 @ group_matrix
        gden = np.maximum(np.minimum(idx, 20), 1)[:, None] * group_matrix.sum(axis=0)[None, :]
        feats.append((gsum / gden)[:, grp]); names.append("번호대_활성도_20회")

        feats.append(rate(bonus_cum, 50)); names.append("보너스출현율_50회")
        feats.append(np.tile(np.arange(1, 46, dtype=np.float32) / 45.0, (T + 1, 1))); names.append("번호크기")

        self.x = np.stack(feats, axis=2).astype(np.float32)
        self.feature_names = names

    # ------------------------------------------------------------------ 학습 / 예측
    @staticmethod
    def _make_models() -> Dict[str, Any]:
        return {
            "gb": HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.04, max_iter=150,
                min_samples_leaf=300, l2_regularization=10.0, random_state=42),
            "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=300, n_jobs=1)),
            "nb": GaussianNB(),
            "rf": RandomForestClassifier(
                n_estimators=80, max_depth=6, min_samples_leaf=200, max_features="sqrt",
                max_samples=0.4, n_jobs=1, random_state=42),
            "dt": DecisionTreeClassifier(max_depth=3, min_samples_leaf=500, random_state=42),
        }

    def fit(self, t_end: int, progress_cb: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        """0 ~ t_end-1 회차(= 1회부터 t_end회까지)로 5개 모델을 각각 학습. progress_cb(0~1)는 학습 시간 비중에 맞춰 보고한다."""
        if self.x is None or self.m is None:
            raise RuntimeError("학습 데이터가 준비되지 않았습니다.")
        t0 = self.MIN_T
        if t_end - t0 < 30:
            raise ValueError("학습 데이터가 부족합니다.")
        F = self.x.shape[2]
        Xtr = self.x[t0:t_end].reshape(-1, F)
        ytr = self.m[t0:t_end].reshape(-1)
        models = {}
        done = 0.0
        for key, est in self._make_models().items():
            est.fit(Xtr, ytr)
            models[key] = est
            done += FIT_WEIGHTS[key]
            if progress_cb:
                progress_cb(min(done, 1.0))
        return {"models": models, "n_rows": int(len(ytr)), "t_end": t_end}

    @staticmethod
    def _norm(p: FloatArray) -> FloatArray:
        """'한 회차에 정확히 6개가 나온다'는 구조를 반영해 확률 합계를 6으로 맞춘다"""
        p = np.clip(p * 6.0 / p.sum(), 0.01, 0.90)
        return p * 6.0 / p.sum()

    def predict(self, fitted: Dict[str, Any], t: int) -> Dict[str, FloatArray]:
        """t번째 회차에서 각 번호가 나올 확률 (모델 5개, 모델이 내놓은 값 그대로)"""
        if self.x is None:
            raise RuntimeError("특징 데이터가 준비되지 않았습니다.")
        x = self.x[t]
        return {k: m.predict_proba(x)[:, 1] for k, m in fitted["models"].items()}

    # ------------------------------------------------------------------ 추천 세트 구성
    def _decorate(self, rank: int, title: str, rule: str, nums: List[int], p: FloatArray, label: str) -> Dict[str, Any]:
        nums = sorted(int(n) for n in nums)
        return {"rank": rank, "title": title, "rule": rule, "nums": nums,
                "metric_title": label,
                "metric": [(n, f"{p[n - 1] * 100:.1f}%") for n in nums]}

    # ------------------------------------------------------------------ 풀
    def pool_at(self, t: int) -> List[int]:
        """t번째 회차를 예측할 때의 풀: 직전 POOL_WINDOW회(t 이전)에 가장 많이 나온 POOL_SIZE개 번호 (동률은 t 로 정한 무작위 순서)"""
        if self.m is None:
            raise RuntimeError("당첨 데이터가 준비되지 않았습니다.")
        counts = self.m[max(0, t - POOL_WINDOW):t].sum(axis=0).astype(np.float64)
        return sorted(int(i) + 1 for i in pick_top(counts, POOL_SIZE, np.random.default_rng(t)))

    # ------------------------------------------------------------------ 모델별 5세트
    def model_sets(self, Pm: Dict[str, FloatArray], t: int, prev: Set[int], pool: List[int]) -> List[Dict[str, Any]]:
        """SET_MODELS 의 모델 5개가 각자 자기 확률로 세트를 만든다. 세트마다 풀(pool) 안에서 구조 조건과 직전 번호 상한(PREV_CAP)을
        모두 통과하는 조합 중 점수 합이 가장 큰 6개를 고른다. 평균하지 않으므로 모델마다의 성격이 그대로 드러난다."""
        in_pool = np.zeros(45, dtype=bool)
        in_pool[[n - 1 for n in pool]] = True
        sets = []
        for rank, (key, title, rule) in enumerate(SET_MODELS, 1):
            score = Pm[key].astype(np.float64).copy()
            prev_idx = [n - 1 for n in prev]
            others = np.flatnonzero(in_pool & ~np.isin(np.arange(45), prev_idx))
            # 직전 번호는 모델의 쏠림을 없애려고, 풀의 다른 번호 점수 중에서 무작위로 뽑은 값을 점수로 쓴다 (시드: 회차·모델 순번)
            score[prev_idx] = np.random.default_rng(1000 * t + rank - 1).choice(score[others], size=len(prev_idx), replace=False)
            nums = pick_best(score, np.random.default_rng(t), prev, in_pool)
            sets.append(self._decorate(rank, title, rule, nums, self._norm(Pm[key]), "확률"))      # 화면에는 모델이 낸 확률(합계 6)을 보여 준다
        return sets

    # ------------------------------------------------------------------ 분석 (다음 회차 추천)
    def sets_at(self, t: int, progress_cb: Optional[Callable[[int], None]] = None) -> List[Dict[str, Any]]:
        """t번째 회차(0부터)를 예측하는 모델별 5세트. t 이전 회차(= t개 회차)만 학습에 쓰므로 t 가 마지막 회차 다음이면
        실제 추천, 그보다 앞이면 과거 되감기가 된다."""
        if self.x is None:
            raise RuntimeError("특징 데이터가 준비되지 않았습니다.")
        if t < self.MIN_T + 100 or t > len(self.past_data):
            raise ValueError(f"학습에는 최소 {self.MIN_T + 100}회 이상의 데이터가 필요합니다.")
        cb = progress_cb or (lambda p: None)

        cb(5)
        prev = set(self.past_data[t - 1])                    # 직전 회차 번호 (점수를 우연 수준으로 맞추는 데 쓴다)
        fitted = self.fit(t, progress_cb=lambda f: cb(5 + int(90 * f)))
        sets = self.model_sets(self.predict(fitted, t), t, prev, self.pool_at(t))
        cb(100)
        return sets

    def analyze(self, progress_cb: Optional[Callable[[int], None]] = None) -> Dict[str, Any]:
        """다음 회차의 모델별 5세트와 무작위 기준 5세트"""
        T = len(self.past_data)
        return {"sets": self.sets_at(T, progress_cb), "random": random_sets(T), "pool": self.pool_at(T), "t": T,
                "round": self.rounds[-1] + 1}


# ==========================================================================
# 브라우저에서 호출하는 함수
# ==========================================================================
_eng: Optional[LottoEngine] = None


def load(rows_json: str) -> int:
    """당첨번호 목록을 불러와 특징을 만든다. 불러온 회차 수를 돌려준다."""
    global _eng
    _eng = LottoEngine()
    _eng.load_rows(json.loads(rows_json))
    return len(_eng.past_data)


def recommend(cb: Optional[Callable[[int], None]] = None) -> str:
    """다음 회차의 모델별 5세트 + 무작위 기준 5세트 + 풀"""
    assert _eng is not None
    res = _eng.analyze(cb)
    return json.dumps({"round": res["round"], "t": res["t"], "sets": res["sets"], "random": res["random"], "pool": res["pool"]}, ensure_ascii=False)


def rewind_step(t: int, cb: Optional[Callable[[int], None]] = None) -> str:
    """t번째 회차(0부터)를 그 직전까지의 데이터로 추천해 실제 당첨번호와 채점"""
    assert _eng is not None
    sets = [s["nums"] for s in _eng.sets_at(t, cb)]
    actual = _eng.past_data[t]
    return json.dumps({"round": _eng.rounds[t], "actual": actual, "ai": match_counts(sets, actual),
                       "rand": match_counts(random_sets(t), actual)}, ensure_ascii=False)
