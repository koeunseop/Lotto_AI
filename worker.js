/* 파이썬 엔진(Pyodide)을 돌리는 작업 스레드: 화면이 멈추지 않도록 학습은 여기서 한다 */
importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js');

let py = null;
let em = null;
let booting = null;

const post = (msg) => self.postMessage(msg);

async function boot() {
  post({ type: 'status', text: '파이썬 엔진 내려받는 중... (처음 한 번만 오래 걸립니다)' });
  py = await loadPyodide();
  post({ type: 'status', text: '수치·학습 라이브러리 불러오는 중...' });
  await py.loadPackage(['numpy', 'scipy', 'scikit-learn']);
  post({ type: 'status', text: '추천 엔진 준비 중...' });
  const src = await (await fetch('engine_mobile.py')).text();
  py.FS.writeFile('engine_mobile.py', src);
  em = py.pyimport('engine_mobile');
  post({ type: 'status', text: '엔진 준비 완료' });
}

self.onmessage = async (e) => {
  const { id, type, payload } = e.data;
  try {
    if (!booting) booting = boot();
    await booting;
    if (type === 'boot') {
      post({ type: 'result', id, data: true });
    } else if (type === 'load') {
      post({ type: 'result', id, data: em.load(JSON.stringify(payload)) });
    } else if (type === 'recommend') {
      const out = em.recommend((p) => post({ type: 'progress', id, pct: p }));
      post({ type: 'result', id, data: JSON.parse(out) });
    } else if (type === 'rewind') {
      const out = em.rewind_step(payload.t, (p) => post({ type: 'progress', id, pct: p }));
      post({ type: 'result', id, data: JSON.parse(out) });
    }
  } catch (err) {
    post({ type: 'error', id, message: String(err && err.message ? err.message : err) });
  }
};
