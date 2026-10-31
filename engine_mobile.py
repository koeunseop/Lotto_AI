# -*- coding: utf-8 -*-
"""
모바일 엔진 (PC 21.0 의 번호 선정 로직을 그대로 옮긴 것: 엑셀/화면 코드 제외)
브라우저(Pyodide) 안에서 실행되며 app.js 가 아래 함수를 호출한다.
"""
import json
from typing import List, Dict, Any, Optional, Callable, Set

import numpy as np
from numpy.typing import NDArray

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_SEED_BASE = 700000                # 무작위 기준선 시드 = RANDOM_SEED_BASE + 학습 회차 수 (하이브리드와 같은 시드 규칙)

FloatArray = NDArray[np.float32]

P_BASE = 6.0 / 45.0

MODEL_KEYS = ["lr", "gb", "rf", "mlp"]   # 기법별 모델 4개 (앙상블 'ens' 는 이 4개의 표준화 평균)

# 관점별 AI: 같은 모델(LR+GB 평균)에 입력 특징만 달리 준다 (features=None 이면 15개 전체)
VIEWS: List[Dict[str, Any]] = [
    {"key": "all", "features": None},
    {"key": "short", "features": ["출현율_1회", "출현율_3회", "출현율_5회", "출현율_10회", "직전_인접수"]},
    {"key": "long", "features": ["출현율_20회", "출현율_50회", "출현율_100회", "출현율_전체", "출현편차_z", "번호크기"]},
    {"key": "gap", "features": ["미출현_간격", "간격/평균간격", "출현율_전체"]},
]

AI_KEY = "hybrid"

PREV_CAP = 1                             # 한 세트에 넣을 수 있는 직전 회차 번호의 최대 개수

# 학습 시간 비중(실측): 진행률을 단계 수가 아니라 실제 걸리는 시간에 맞춰 보고하기 위한 값
FIT_WEIGHTS = {"lr": 0.01, "gb": 0.27, "rf": 0.07, "mlp": 0.65}   # 기법별 모델
VIEW_WEIGHTS = [0.84, 0.05, 0.05, 0.06]                           # 관점별 AI (VIEWS 순서)
VIEW_SHARE = 0.1                                                  # 전체 학습 시간 중 관점별 AI 비중 (나머지는 기법별 모델)


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


def cap_sample(nums: List[int], p: NDArray[np.floating[Any]], prev: Set[int], cap: int,
               rng: np.random.Generator) -> List[int]:
    """가중 추첨 세트에서 직전 회차 번호가 cap개를 넘으면 확률이 낮은 것부터 빼고, 직전 번호가 아닌 번호를 같은 가중 추첨으로 채운다."""
    extra = sum(n in prev for n in nums) - cap
    if extra <= 0:
        return list(nums)
    drop = sorted((n for n in nums if n in prev), key=lambda n: p[n - 1])[:extra]
    kept = [n for n in nums if n not in drop]
    pool = [n for n in range(1, 46) if n not in kept and n not in prev]
    w = p[np.asarray(pool) - 1]
    return kept + [int(n) for n in rng.choice(pool, size=len(drop), replace=False, p=w / w.sum())]


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

        grp = np.arange(45) // 10
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
            "lr": make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500)),
            "gb": HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.04, max_iter=150,
                min_samples_leaf=300, l2_regularization=10.0, random_state=42),
            "rf": RandomForestClassifier(
                n_estimators=80, max_depth=6, min_samples_leaf=200, max_features="sqrt",
                max_samples=0.4, n_jobs=1, random_state=42),
            # 조기종료는 정확도 기준이라 양성 13% 데이터에서 8회 만에 멈추므로 끄고, 반복 횟수·규제를 고정한다
            "mlp": make_pipeline(StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(32, 16), alpha=1.0, learning_rate_init=0.001, max_iter=25,
                early_stopping=False, random_state=42)),
        }

    def fit(self, t_end: int, progress_cb: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        """0 ~ t_end-1 회차(= 1회부터 t_end회까지)로 4개 기법을 각각 학습. progress_cb(0~1)는 학습 시간 비중에 맞춰 보고한다."""
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
        """t번째 회차에서 각 번호가 나올 확률 (기법별 + 앙상블)"""
        if self.x is None:
            raise RuntimeError("특징 데이터가 준비되지 않았습니다.")
        x = self.x[t]
        P = {k: self._norm(m.predict_proba(x)[:, 1]) for k, m in fitted["models"].items()}
        # 앙상블: 모델마다 표준화한 점수를 평균 (확률 변동폭이 큰 모델이 결과를 독점하지 않도록 4개 모델을 동등 반영)
        z = np.mean([(P[k] - P[k].mean()) / (P[k].std() + 1e-9) for k in MODEL_KEYS], axis=0)
        sd = np.mean([P[k].std() for k in MODEL_KEYS])
        P["ens"] = self._norm(P_BASE + sd * z)
        return P

    # ------------------------------------------------------------------ 추천 세트 구성
    def _decorate(self, rank: int, title: str, rule: str, nums: List[int], p: FloatArray, label: str) -> Dict[str, Any]:
        nums = sorted(int(n) for n in nums)
        return {"rank": rank, "title": title, "rule": rule, "nums": nums,
                "metric_title": label,
                "metric": [(n, f"{p[n - 1] * 100:.1f}%") for n in nums]}

    # ------------------------------------------------------------------ 관점별 AI 4개
    def fit_views(self, t_end: int, progress_cb: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        """1~t_end회로 관점별 AI(LR+GB) 4개를 학습한다. progress_cb(0~1)는 학습 시간 비중에 맞춰 보고한다."""
        if self.x is None or self.m is None:
            raise RuntimeError("학습 데이터가 준비되지 않았습니다.")
        t0 = self.MIN_T
        if t_end - t0 < 30:
            raise ValueError("학습 데이터가 부족합니다.")
        ytr = self.m[t0:t_end].reshape(-1)
        out: Dict[str, Any] = {}
        done = 0.0
        for n, v in enumerate(VIEWS):
            idx = (list(range(len(self.feature_names))) if v["features"] is None
                   else [self.feature_names.index(f) for f in v["features"]])
            Xtr = self.x[t0:t_end][:, :, idx].reshape(-1, len(idx))
            lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500))
            lr.fit(Xtr, ytr)
            gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=150,
                                                min_samples_leaf=300, l2_regularization=10.0, random_state=42)
            gb.fit(Xtr, ytr)
            out[v["key"]] = (idx, lr, gb)
            done += VIEW_WEIGHTS[n] if n < len(VIEW_WEIGHTS) else 1.0 / len(VIEWS)
            if progress_cb:
                progress_cb(min(done, 1.0))
        return out

    def predict_views(self, models: Dict[str, Any], t: int) -> Dict[str, FloatArray]:
        """t번째 회차의 번호별 출현 확률 (관점별, 합계 6)"""
        if self.x is None:
            raise RuntimeError("특징 데이터가 준비되지 않았습니다.")
        P: Dict[str, FloatArray] = {}
        for k, (idx, lr, gb) in models.items():
            x = self.x[t][:, idx]
            p = 0.5 * (lr.predict_proba(x)[:, 1] + gb.predict_proba(x)[:, 1])
            p = np.clip(p * 6.0 / p.sum(), 0.01, 0.90)
            P[k] = p * 6.0 / p.sum()
        return P

    # ------------------------------------------------------------------ 하이브리드 5세트
    @classmethod
    def _blend(cls, ps: List[FloatArray]) -> FloatArray:
        """확률 벡터 여러 개를 표준화해 평균한다 (변동폭이 큰 쪽이 결과를 독점하지 않도록 동등 반영)"""
        z = np.mean([(p - p.mean()) / (p.std() + 1e-9) for p in ps], axis=0)
        sd = np.mean([p.std() for p in ps])
        return cls._norm(P_BASE + sd * z)

    def hybrid_sets(self, Pm: Dict[str, FloatArray], Pv: Dict[str, FloatArray], t: int, prev: Set[int]) -> List[Dict[str, Any]]:
        """기법별 모델 4개 + 관점별 AI 4개의 확률로 5세트를 만든다. 직전 회차 번호는 세트당 PREV_CAP개까지만 넣는다."""
        sources = [Pm[k] for k in MODEL_KEYS] + [Pv[v["key"]] for v in VIEWS]
        p_hyb = self._blend(sources)
        p_view = self._blend([Pv[v["key"]] for v in VIEWS])
        p_model = Pm["ens"]

        def top(p: FloatArray) -> List[int]:
            return (pick_top_capped(p, 6, np.random.default_rng(t), prev, PREV_CAP) + 1).tolist()

        votes = np.zeros(45)
        for p in sources:
            for i in pick_top_capped(p, 6, np.random.default_rng(t), prev, PREV_CAP):
                votes[i] += 1
        draw = (np.random.default_rng(900000 + t).choice(45, size=6, replace=False, p=p_hyb / p_hyb.sum()) + 1).tolist()
        draw = cap_sample(draw, p_hyb, prev, PREV_CAP, np.random.default_rng(900001 + t))

        return [
            self._decorate(1, "하이브리드 종합", "8개 AI의 확률을 표준화해 평균 · 상위 6개", top(p_hyb), p_hyb, "하이브리드 확률"),
            self._decorate(2, "관점 평균", "관점별 AI 4개(종합·단기·장기·간격)의 평균 확률 · 상위 6개", top(p_view), p_view, "관점 평균 확률"),
            self._decorate(3, "기법 평균", "기법별 모델 4개(로지스틱·GB·RF·신경망)의 평균 확률 · 상위 6개", top(p_model), p_model, "기법 평균 확률"),
            self._decorate(4, "확률 추첨", "하이브리드 확률대로 가중 추첨 (확률 차이가 작아 사실상 무작위에 가까움)", draw, p_hyb, "하이브리드 확률"),
            self._decorate(5, "8개 AI 득표", "8개 AI가 각자 뽑은 상위 6개에 많이 오른 번호 순 · 동률은 하이브리드 확률", top(votes + p_hyb), p_hyb, "하이브리드 확률"),
        ]

    # ------------------------------------------------------------------ 분석 (다음 회차 추천)
    def sets_at(self, t: int, progress_cb: Optional[Callable[[int], None]] = None) -> List[Dict[str, Any]]:
        """t번째 회차(0부터)를 예측하는 하이브리드 5세트. t 이전 회차(= t개 회차)만 학습에 쓰므로 t 가 마지막 회차 다음이면
        실제 추천, 그보다 앞이면 과거 되감기가 된다."""
        if self.x is None:
            raise RuntimeError("특징 데이터가 준비되지 않았습니다.")
        if t < self.MIN_T + 100 or t > len(self.past_data):
            raise ValueError(f"AI 학습에는 최소 {self.MIN_T + 100}회 이상의 데이터가 필요합니다.")
        cb = progress_cb or (lambda p: None)

        cb(5)
        prev = set(self.past_data[t - 1])                    # 직전 회차 번호 (세트당 PREV_CAP개까지만 사용)
        views = self.fit_views(t, progress_cb=lambda f: cb(5 + int(90 * VIEW_SHARE * f)))
        fitted = self.fit(t, progress_cb=lambda f: cb(5 + int(90 * (VIEW_SHARE + (1 - VIEW_SHARE) * f))))
        sets = self.hybrid_sets(self.predict(fitted, t), self.predict_views(views, t), t, prev)
        cb(100)
        return sets

    def analyze(self, progress_cb: Optional[Callable[[int], None]] = None) -> Dict[str, Any]:
        """다음 회차의 하이브리드 5세트와 무작위 기준 5세트"""
        T = len(self.past_data)
        return {"sets": self.sets_at(T, progress_cb), "random": random_sets(T), "t": T, "round": self.rounds[-1] + 1}


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
    """다음 회차의 하이브리드 5세트 + 무작위 기준 5세트"""
    assert _eng is not None
    res = _eng.analyze(cb)
    return json.dumps({"round": res["round"], "t": res["t"], "sets": res["sets"], "random": res["random"]}, ensure_ascii=False)


def rewind_step(t: int, cb: Optional[Callable[[int], None]] = None) -> str:
    """t번째 회차(0부터)를 그 직전까지의 데이터로 추천해 실제 당첨번호와 채점"""
    assert _eng is not None
    sets = [s["nums"] for s in _eng.sets_at(t, cb)]
    actual = _eng.past_data[t]
    return json.dumps({"round": _eng.rounds[t], "actual": actual, "ai": match_counts(sets, actual),
                       "rand": match_counts(random_sets(t), actual)}, ensure_ascii=False)
