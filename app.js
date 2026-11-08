'use strict';
/* LOTTO AI 모바일: 화면·저장·채점은 여기서, 번호 선정(파이썬 엔진)은 worker.js 에서 한다 */

const STORE_KEY = 'lotto.state.v1';
const RULE = '21.1';                        // 번호 선정 규칙 버전: 다른 규칙의 기록은 성적에 섞지 않는다
const EXPECTED = 0.8;                       // 무작위 세트 1개의 기대 일치 개수
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

let draws = [];                             // [회차, 번호1..번호6, 보너스]
let records = {};                           // 회차 -> {sets, random, detail}  (추천 기록)
let busy = false;
let loadedKey = '';
let rewindStop = false;

/* ---------------------------------------------------------------- 저장 */
function persist() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify({ draws, records })); } catch (e) { /* 저장 불가 환경: 이번 실행에서만 유지 */ }
}

async function loadState() {
  try {
    const s = JSON.parse(localStorage.getItem(STORE_KEY) || 'null');
    if (s && Array.isArray(s.draws) && s.draws.length) { draws = s.draws; records = s.records || {}; return; }
  } catch (e) { /* 저장본이 없거나 깨짐 -> 기본 데이터 사용 */ }
  draws = await (await fetch('seed_draws.json')).json();
  records = {};
  persist();
}

const lastRound = () => draws[draws.length - 1][0];
const drawKey = () => `${draws.length}/${draws[draws.length - 1].join(',')}`;

/* ---------------------------------------------------------------- 공 그리기 */
function ballColor(n) {
  if (n <= 10) return ['#FFC107', '#212529'];
  if (n <= 20) return ['#1976D2', '#fff'];
  if (n <= 30) return ['#D32F2F', '#fff'];
  if (n <= 40) return ['#616161', '#fff'];
  return ['#388E3C', '#fff'];
}
function balls(nums, hit) {
  return '<div class="balls">' + nums.map((n) => {
    const [bg, fg] = ballColor(n);
    return `<span class="ball${hit && hit.has(n) ? ' hit' : ''}" style="background:${bg};color:${fg}">${n}</span>`;
  }).join('') + '</div>';
}

/* ---------------------------------------------------------------- 엔진(worker) */
let worker = null;
const waiting = new Map();
let seq = 0;
let onEngineStatus = null;

/* 작업 스레드는 처음 필요할 때 만든다: 서비스 워커가 켜진 뒤에 만들어야 내려받은 엔진이 오프라인용으로 저장된다 */
async function getWorker() {
  if (worker) return worker;
  if ('serviceWorker' in navigator) {
    try {
      await Promise.race([navigator.serviceWorker.ready, new Promise((r) => setTimeout(r, 5000))]);
      for (let i = 0; i < 20 && !navigator.serviceWorker.controller; i++) await new Promise((r) => setTimeout(r, 100));
    } catch (e) { /* 서비스 워커 없이도 동작 */ }
  }
  worker = new Worker('worker.js');
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'status') { if (onEngineStatus) onEngineStatus(m.text); return; }
    const w = waiting.get(m.id);
    if (!w) return;
    if (m.type === 'progress') { if (w.onProgress) w.onProgress(m.pct); return; }
    waiting.delete(m.id);
    if (m.type === 'result') w.resolve(m.data); else w.reject(new Error(m.message));
  };
  worker.onerror = (e) => { for (const w of waiting.values()) w.reject(new Error(e.message || '엔진 오류')); waiting.clear(); };
  return worker;
}

async function call(type, payload, onProgress) {
  const w = await getWorker();
  return new Promise((resolve, reject) => {
    const id = ++seq;
    waiting.set(id, { resolve, reject, onProgress });
    w.postMessage({ id, type, payload });
  });
}

async function ensureLoaded() {
  if (loadedKey === drawKey()) return;
  await call('load', draws);
  loadedKey = drawKey();
}

/* ---------------------------------------------------------------- 채점 */
const matches = (sets, actual) => { const a = new Set(actual); return sets.map((s) => s.filter((n) => a.has(n)).length); };
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

function scoredRows() {
  const byRound = new Map(draws.map((d) => [d[0], d]));
  const rows = [];
  for (const key of Object.keys(records).map(Number).sort((a, b) => a - b)) {
    const d = byRound.get(key);
    if (!d) continue;
    const rec = records[key];
    if (rec.rule !== RULE) continue;
    const actual = d.slice(1, 7);
    rows.push({ round: key, actual, ai: matches(rec.sets, actual), rand: matches(rec.random, actual), rec });
  }
  return rows;
}

function summary(rows) {
  if (!rows.length) return '';
  const ai = rows.map((r) => mean(r.ai));
  const rd = rows.map((r) => mean(r.rand));
  const win = ai.filter((a, i) => a > rd[i] + 1e-9).length;
  const lose = ai.filter((a, i) => a < rd[i] - 1e-9).length;
  const hit3 = rows.reduce((s, r) => s + r.ai.filter((m) => m >= 3).length, 0);
  const hit3r = rows.reduce((s, r) => s + r.rand.filter((m) => m >= 3).length, 0);
  return `하이브리드 ${rows.length}회 · 세트당 평균 일치 ${mean(ai).toFixed(2)}개 (같은 회차 무작위 ${mean(rd).toFixed(2)}개)\n` +
    `회차별 무작위와 비교 ${win}승 ${rows.length - win - lose}무 ${lose}패\n` +
    `3개 이상 맞힌 세트 ${hit3}개 (무작위 ${hit3r}개)\n` +
    `무작위 5세트의 기대 평균은 세트당 ${EXPECTED.toFixed(2)}개입니다. 회차가 적으면 승패 차이는 우연일 수 있습니다(대략 50회 이상 쌓여야 가릴 수 있습니다).`;
}

function resultCell(a, r) { return a > r + 1e-9 ? ['승', 'win'] : a < r - 1e-9 ? ['패', 'lose'] : ['무', '']; }

function scoreTable(rows, withDetail) {
  if (!rows.length) return '';
  let h = '';
  for (const r of rows.slice().reverse()) {
    const a = mean(r.ai), b = mean(r.rand), [txt, cls] = resultCell(a, b);
    const head = `<table><tr><td style="width:25%">${r.round}회</td><td style="width:25%">${a.toFixed(1)}</td><td style="width:25%">${b.toFixed(1)}</td><td class="${cls}" style="width:25%">${txt}</td></tr></table>`;
    if (!withDetail) { h += head; continue; }
    const hit = new Set(r.actual);
    const block = (sets, ms, names) => sets.map((s, i) =>
      `<div class="set"><div class="ttl">${esc(names[i])} · ${ms[i]}개 일치</div>${balls(s, hit)}</div>`).join('');
    const aiNames = r.rec.detail.map((d) => d.title);
    h += `<details><summary>${head}</summary>
      <div class="muted">당첨번호</div>${balls(r.actual)}
      <div class="muted">하이브리드</div>${block(r.rec.sets, r.ai, aiNames)}
      <div class="muted">무작위</div>${block(r.rec.random, r.rand, r.rand.map((_, i) => `무작위 ${i + 1}`))}
    </details>`;
  }
  return '<table><tr><th style="width:25%">회차</th><th style="width:25%">하이브리드</th><th style="width:25%">무작위</th><th style="width:25%">결과</th></tr></table>' + h;
}

/* ---------------------------------------------------------------- 화면 */
function renderHeader() {
  const d = draws[draws.length - 1];
  $('#last').textContent = `최근 ${d[0]}회 당첨 : ${d.slice(1, 7).join(', ')} + 보너스 ${d[7]}`;
  $('#go').textContent = `추천 받기 (${d[0] + 1}회)`;
}

function renderResult(round, detail, sets, random, note) {
  const used = [...new Set(sets.flat())].sort((a, b) => a - b);
  const usedR = [...new Set(random.flat())].sort((a, b) => a - b);
  const aiHtml = sets.map((nums, i) => {
    const d = detail[i];
    const prob = d.metric && d.metric.length ? `<div class="prob">${esc(d.mt)}: ${d.metric.map((m) => `${m[0]} ${esc(m[1])}`).join(' · ')}</div>` : '';
    return `<div class="set"><div class="ttl">[SET ${i + 1}] ${esc(d.title)}</div>${balls(nums)}<div class="rule">${esc(d.rule)}</div>${prob}</div>`;
  }).join('');
  const rdHtml = random.map((nums, i) => `<div class="set"><div class="ttl">[SET ${i + 1}] 무작위 ${i + 1}</div>${balls(nums)}</div>`).join('');
  const both = used.filter((n) => usedR.includes(n));
  $('#result').innerHTML =
    `<div class="card"><h2>${round}회 하이브리드</h2><div class="muted">${esc(note)}</div>${aiHtml}
       <div class="used">사용 번호 ${used.length}개: ${used.join(', ')}</div></div>
     <div class="card"><h2>${round}회 무작위 기준</h2><div class="muted">순수 무작위 6개 5세트 (직전 번호 제한 없음)</div>${rdHtml}
       <div class="used">사용 번호 ${usedR.length}개: ${usedR.join(', ')}</div></div>
     <div class="muted" style="margin:0 4px 12px">둘 다 사용한 번호 ${both.length}개: ${both.join(', ') || '없음'}</div>`;
}

function renderSaved() {
  const rec = records[lastRound() + 1];
  if (rec && rec.rule === RULE) renderResult(lastRound() + 1, rec.detail, rec.sets, rec.random, '저장된 추천입니다. 새 회차가 입력되기 전에는 같은 번호가 나옵니다.');
  else $('#result').innerHTML = '';
}

function renderScore() {
  const rows = scoredRows();
  $('#sum').textContent = rows.length ? summary(rows) : '아직 채점된 기록이 없습니다.';
  const old = Object.keys(records).map(Number).filter((r) => records[r].rule !== RULE && r <= lastRound()).length;
  if (old) $('#sum').textContent += `
이전 규칙(v21.0 이하)으로 만든 기록 ${old}회는 성적에서 제외됩니다.`;
  const pending = Object.keys(records).map(Number).filter((r) => r > lastRound() && records[r].rule === RULE);
  if (pending.length) $('#sum').textContent += `\n채점 대기: ${pending.join('회, ')}회 (해당 회차를 입력하면 자동 채점)`;
  $('#scoreRows').innerHTML = scoreTable(rows, true);
  const avail = draws.length - 150;
  $('#rwMax').textContent = `회차 (최대 ${Math.max(avail, 0)})`;
  $('#rwN').max = Math.max(avail, 1);
}

function renderData() {
  $('#inRound').value = lastRound() + 1;
  const small = draws.slice(-8).reverse().map((d) =>
    `<div class="set"><div class="ttl">${d[0]}회</div>${balls(d.slice(1, 7))}<div class="muted">보너스 ${d[7]}</div></div>`).join('');
  $('#recent').innerHTML = small;
}

function renderAll() { renderHeader(); renderSaved(); renderScore(); renderData(); }

function setBusy(v) {
  busy = v;
  for (const id of ['#go', '#rwGo', '#addDraw', '#delLast', '#imp']) $(id).disabled = v;
}

function setBar(pct) {
  $('#bar').style.display = pct == null ? 'none' : 'block';
  if (pct != null) $('#barfill').style.width = `${pct}%`;
}

/* ---------------------------------------------------------------- 추천 */
async function recommend() {
  if (busy) return;
  setBusy(true);
  setBar(3);
  $('#status').style.color = 'var(--blue)';
  onEngineStatus = (t) => { $('#status').textContent = t; };
  const t0 = Date.now();
  try {
    $('#status').textContent = '엔진 준비 중...';
    await ensureLoaded();
    $('#status').textContent = '학습 중...';
    const res = await call('recommend', null, (p) => { setBar(p); $('#status').textContent = `학습 중 ${p}%`; });
    const detail = res.sets.map((s) => ({ title: s.title, rule: s.rule, mt: s.metric_title, metric: s.metric }));
    records[res.round] = { sets: res.sets.map((s) => s.nums), random: res.random, detail, rule: RULE };
    persist();
    renderResult(res.round, detail, records[res.round].sets, res.random, `직전 회차 번호 세트당 최대 1개 · 무작위 기준과 함께 기록됨 (${((Date.now() - t0) / 1000).toFixed(0)}초)`);
    renderScore();
    $('#status').style.color = 'var(--green)';
    $('#status').textContent = `${res.round}회 추천 완료`;
  } catch (e) {
    $('#status').style.color = 'var(--red)';
    $('#status').textContent = `오류: ${e.message}`;
  } finally {
    onEngineStatus = null;
    setBar(null);
    setBusy(false);
  }
}

/* ---------------------------------------------------------------- 되감기 */
async function rewind() {
  if (busy) return;
  const T = draws.length;
  const avail = T - 150;
  if (avail < 1) { $('#rwStatus').textContent = '되감을 수 있는 회차가 없습니다.'; return; }
  const n = Math.max(1, Math.min(avail, parseInt($('#rwN').value, 10) || 5));
  setBusy(true);
  rewindStop = false;
  $('#rwStop').disabled = false;
  $('#rwRows').innerHTML = '';
  $('#rwSum').textContent = '';
  onEngineStatus = (t) => { $('#rwStatus').textContent = t; };
  const rows = [];
  try {
    await ensureLoaded();
    for (let i = 1, t = T - n; t < T; t++, i++) {
      if (rewindStop) break;
      $('#rwStatus').textContent = `${i}/${n}회 진행 중 · ${draws[t][0]}회 학습·추천...`;
      const r = await call('rewind', { t });
      rows.push({ round: r.round, actual: r.actual, ai: r.ai, rand: r.rand });
      $('#rwRows').innerHTML = scoreTable(rows, false);
      $('#rwSum').textContent = summary(rows);
    }
    $('#rwStatus').textContent = (rewindStop ? '중단됨 · ' : '완료 · ') + `${rows.length}회 채점`;
  } catch (e) {
    $('#rwStatus').textContent = `오류: ${e.message}`;
  } finally {
    onEngineStatus = null;
    $('#rwStop').disabled = true;
    setBusy(false);
  }
}

/* ---------------------------------------------------------------- 데이터 입력·백업 */
function addDraw() {
  const round = parseInt($('#inRound').value, 10);
  const vals = [...document.querySelectorAll('#inNums input')].map((i) => parseInt(i.value, 10));
  const nums = vals.slice(0, 6), bonus = vals[6];
  if (!Number.isInteger(round) || round < 1) return alert('회차를 입력하세요.');
  if (vals.some((v) => !Number.isInteger(v) || v < 1 || v > 45)) return alert('번호는 1~45 사이 숫자 7개(번호 6개 + 보너스)여야 합니다.');
  if (new Set(nums).size !== 6) return alert('번호 6개가 서로 달라야 합니다.');
  if (nums.includes(bonus)) return alert('보너스는 번호 6개와 달라야 합니다.');
  const idx = draws.findIndex((d) => d[0] === round);
  if (idx < 0 && round !== lastRound() + 1) return alert(`다음 회차는 ${lastRound() + 1}회입니다. 회차를 확인하세요.`);
  if (idx >= 0 && !confirm(`${round}회가 이미 있습니다. 바꿀까요?`)) return;
  const row = [round, ...nums.slice().sort((a, b) => a - b), bonus];
  if (idx >= 0) draws[idx] = row; else draws.push(row);
  persist();
  document.querySelectorAll('#inNums input').forEach((i) => { i.value = ''; });
  renderAll();
  alert(`${round}회를 저장했습니다.`);
}

function delLast() {
  if (draws.length <= 151) return alert('더 이상 삭제할 수 없습니다.');
  if (!confirm(`${lastRound()}회를 삭제할까요?`)) return;
  draws.pop();
  persist();
  renderAll();
}

function exportData() {
  const blob = new Blob([JSON.stringify({ version: 1, draws, records })], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `lotto_backup_${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

async function importData(file) {
  try {
    const s = JSON.parse(await file.text());
    const ok = Array.isArray(s.draws) && s.draws.length > 150 && s.draws.every((d) => Array.isArray(d) && d.length === 8 && d.every(Number.isInteger));
    if (!ok) return alert('올바른 백업 파일이 아닙니다.');
    if (!confirm(`현재 데이터를 백업 파일(${s.draws[s.draws.length - 1][0]}회까지)로 바꿀까요?`)) return;
    draws = s.draws;
    records = s.records || {};
    persist();
    renderAll();
    alert('가져왔습니다.');
  } catch (e) {
    alert('파일을 읽지 못했습니다.');
  }
}

/* ---------------------------------------------------------------- 시작 */
function setup() {
  const box = $('#inNums');
  ['1', '2', '3', '4', '5', '6', '보너스'].forEach((p) => {
    box.insertAdjacentHTML('beforeend', `<input type="number" inputmode="numeric" min="1" max="45" placeholder="${p}">`);
  });
  document.querySelectorAll('nav button').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('nav button').forEach((x) => x.classList.toggle('on', x === b));
    document.querySelectorAll('section').forEach((s) => s.classList.toggle('on', s.id === `tab-${b.dataset.tab}`));
    window.scrollTo(0, 0);
  }));
  $('#go').addEventListener('click', recommend);
  $('#rwGo').addEventListener('click', rewind);
  $('#rwStop').addEventListener('click', () => { rewindStop = true; $('#rwStatus').textContent = '중단하는 중... (진행 중인 회차까지만 마칩니다)'; });
  $('#addDraw').addEventListener('click', addDraw);
  $('#delLast').addEventListener('click', delLast);
  $('#exp').addEventListener('click', exportData);
  $('#imp').addEventListener('click', () => $('#impFile').click());
  $('#impFile').addEventListener('change', (e) => { if (e.target.files[0]) importData(e.target.files[0]); e.target.value = ''; });

  let installEvt = null;
  window.addEventListener('beforeinstallprompt', (e) => { e.preventDefault(); installEvt = e; $('#install').style.display = 'block'; });
  $('#install').addEventListener('click', async () => { if (installEvt) { installEvt.prompt(); installEvt = null; $('#install').style.display = 'none'; } });

  if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => { /* 오프라인 저장 없이도 동작 */ });
}

setup();
loadState().then(renderAll).catch((e) => { $('#last').textContent = `데이터를 불러오지 못했습니다: ${e.message}`; });
