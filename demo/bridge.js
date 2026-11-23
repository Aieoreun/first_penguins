/* 시연용 연결 코드. 프로토타입의 채팅 입력을 실제 /tutor 서버로 보낸다.
   프로토타입 파일은 고치지 않고, 중계 서버(serve.py)가 이 파일을 끼워 넣는다. */
(() => {
  const ACTIONS = {
    answer: '답변', scaffold: '힌트로 유도', clarify: '되묻기', abstain: '교재에 없음',
    redirect: '수업으로 되돌림', refuse: '거절', escalate: '도움 받을 곳 안내',
  };
  const SUBJECTS = { finance: '금융', python: '파이썬', prompt: '프롬프트' };
  // 시연은 질문을 하나씩 따로 보낸다. 화면에 앞 대화가 남아 있어도 서버에는 지금 질문만 간다.
  let busy = false;

  const style = document.createElement('style');
  style.textContent = `
    .live-action{display:inline-block;font-size:10px;letter-spacing:.3px;padding:2px 9px;border-radius:99px;background:#eaf3ff;color:#0878f9;margin-bottom:9px}
    .live-action.stop{background:#fdecec;color:#c0392b}
    .live-action.soft{background:#fff4e0;color:#a05a00}
    .live-answer p{white-space:pre-wrap}
    .live-section{white-space:pre-wrap;border-left:2px solid #73a4ed;background:#f7faff;padding:10px 13px;margin-top:8px;border-radius:0 6px 6px 0;font-size:12px;max-height:260px;overflow:auto}
    .live-wait{color:#9ea7b4}
    /* 긴 답변은 앞부분만 보여 주고 펼쳐 보게 한다. 서버가 준 답변 글은 그대로다 */
    .live-answer.folded{max-height:154px;overflow:hidden;position:relative}
    .live-answer.folded::after{content:'';position:absolute;left:0;right:0;bottom:0;height:44px;background:linear-gradient(transparent,#fff)}
    .live-more{border:0;background:none;color:#0878f9;font-size:12px;padding:4px 0 0;cursor:pointer}
    /* 시연에서 쓰지 않는 것: 예시 자동 재생, 튜터 말투 고르기 */
    .tour-start-button, select.tone, .assistant-top small{display:none !important}
    /* 시연은 질의응답과 교재 함께 보기만 쓴다. 나머지 예시 화면 요소는 숨긴다 */
    .nav-item[data-panel="map"], .nav-item[data-action="subject-graph"], .nav-item[data-panel="records"],
    .history-group, .sidebar-foot, .suggestions, .resume-line, .demo-note, .suggested-followups,
    .context-chip, #course-context, .chat-header, #map-toggle{display:none !important}
    .book-pick{width:100%;padding:9px 10px;border:1px solid #dfe5ee;border-radius:8px;background:#fff;font-size:12px;color:#3d4756}
    .book-legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:#6b7686;margin-top:12px}
    .book-legend i{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:5px}
    .book-legend .lq{background:#fff1b8}
    .book-legend .lf{background:#fff;border-left:3px solid #0878f9;border-radius:0}
    .book-paper h4{font-size:15px;margin:22px 0 10px}
    .book-paper h5{font-size:13px;margin:16px 0 8px;color:#2c3543}
    .book-paper p, .book-paper li{font-size:13px;line-height:1.9;color:#4b5360}
    .book-paper pre{background:#f4f6fa;border-radius:6px;padding:10px 12px;font-size:12px;overflow:auto;white-space:pre}
    .book-paper code{background:#f1f3f7;border-radius:3px;padding:0 3px;font-size:12px}
    .book-paper pre code{background:none;padding:0}
    .book-paper table{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0}
    .book-paper th, .book-paper td{border:1px solid #e1e7ef;padding:5px 8px;text-align:left;color:#4b5360}
    .book-paper th{background:#f7f9fc}
    .book-paper mark, .book-paper .quoted{background:#fff1b8;color:inherit;border-radius:2px}
    .book-paper .quoted.focus{padding-top:4px;padding-bottom:4px;margin-top:0;margin-bottom:0}
    .book-paper p.quoted.focus{padding-bottom:12px}
    .book-paper .focus{border-left:3px solid #0878f9;padding-left:11px;margin-left:-14px}`;
  document.head.appendChild(style);

  const inline = s => escapeHTML(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`([^`]+)`/g, '<code>$1</code>');
  function render(text) {
    return text.split(/\n{2,}/).map(block => '<p>' + block.split('\n').map(line => {
      const heading = line.match(/^#{1,4}\s+(.*)$/);
      return heading ? `<strong>${inline(heading[1])}</strong>` : inline(line);
    }).join('\n') + '</p>').join('');
  }
  const tone = action => ['refuse', 'escalate'].includes(action) ? ' stop' : ['abstain', 'redirect', 'clarify'].includes(action) ? ' soft' : '';

  // ---- 교재 함께 보기 ----
  const book = { toc: null, id: null, answer: '', question: '' };
  const norm = s => String(s).replace(/[`*#>|\-\s]/g, '');
  const getJSON = async url => (await fetch(url)).json();

  function blocksOf(data) {
    const blocks = [];
    let fence = null;
    let para = [];
    const flush = () => { if (para.length) { blocks.push({ type: 'p', text: para.join(' ') }); para = []; } };
    for (const line of data.text.split('\n')) {
      if (fence) {
        if (line.startsWith('```')) { blocks.push({ type: 'code', text: fence.join('\n') }); fence = null; } else fence.push(line);
        continue;
      }
      const table = line.match(/^\(표:\s*(.+?)\)$/);
      const heading = line.match(/^(#{2,4})\s+(.*)$/);
      if (line.startsWith('```')) { flush(); fence = []; }
      else if (table) { flush(); blocks.push({ type: 'table', rows: data.tables[table[1]] || [] }); }
      else if (heading) { flush(); blocks.push({ type: heading[1].length === 2 ? 'h4' : 'h5', text: heading[2] }); }
      else if (/^[-*]\s+/.test(line)) { flush(); blocks.push({ type: 'li', text: line.replace(/^[-*]\s+/, '') }); }
      else if (!line.trim()) flush();
      else para.push(line.trim());
    }
    flush();
    return blocks;
  }

  // 답변에 글자 그대로 들어 있는 문장을 칠한다. 문장 수와 칠한 문장 수를 함께 돌려준다.
  function paint(text, answer, count) {
    return text.split(/(?<=[.!?])\s+/).map(sentence => {
      const n = norm(sentence);
      count.all++;
      if (n.length >= 6 && answer.includes(n)) { count.hit++; return `<mark>${inline(sentence)}</mark>`; }
      return inline(sentence);
    }).join(' ');
  }

  function renderSection(data) {
    const answer = norm(book.answer);
    const blocks = blocksOf(data);
    const count = { all: 0, hit: 0 };
    const html = blocks.map((b, i) => {
      if (b.type === 'p') return `<p data-b="${i}">${paint(b.text, answer, count)}</p>`;
      if (b.type === 'li') return `<ul><li data-b="${i}">${paint(b.text, answer, count)}</li></ul>`;
      if (b.type === 'code') {
        const hit = answer && answer.includes(norm(b.text));
        return `<pre data-b="${i}"${hit ? ' class="quoted"' : ''}><code>${escapeHTML(b.text)}</code></pre>`;
      }
      if (b.type === 'table') {
        const [head, ...rows] = b.rows;
        if (!head) return '';
        const row = (cells, tag) => `<tr>${cells.map(c => `<${tag}>${escapeHTML(c)}</${tag}>`).join('')}</tr>`;
        return `<table data-b="${i}">${row(head, 'th')}${rows.map(r => row(r, 'td')).join('')}</table>`;
      }
      return `<${b.type} data-b="${i}">${inline(b.text)}</${b.type}>`;
    });
    // 섹션이 통째로 인용된 답이면 전부 칠하지 않는다. 서버는 소단원을 관련도 순으로 줄 세워 인용하므로,
    // 답변 맨 앞에 온 소단원(서버가 질문에 가장 가깝다고 본 곳)만 칠한다. 화면에서 따로 고르지 않는다.
    const whole = count.all > 0 && count.hit / count.all >= 0.6;
    let first = [];
    if (whole) {
      const lead = book.answer.match(/^##\s+(.+)$/m);
      const name = lead ? norm(lead[1]) : '';
      if (name === '표') first = blocks.map((b, k) => (b.type === 'table' ? k : -1)).filter(k => k >= 0);
      else {
        const start = blocks.findIndex(b => b.type === 'h4' && norm(b.text) === name);
        if (start >= 0) {
          for (let k = start + 1; k < blocks.length && blocks[k].type !== 'h4'; k++) if (blocks[k].type !== 'table') first.push(k);
          first.unshift(start);
        }
      }
    }
    const out = whole ? html.join('').replace(/<\/?mark>/g, '').replace(/ class="quoted"/g, '') : html.join('');
    return { html: out, first, hit: whole ? 0 : count.hit, whole };
  }

  async function drawBook() {
    const body = $('panel-body');
    $('panel-title').innerHTML = icon('book') + '교재 함께 보기';
    if (!book.toc) { body.innerHTML = '<p>교재를 불러오는 중이에요…</p>'; book.toc = await getJSON('/api/toc'); }
    if (!book.id) book.id = book.toc[0].sections[0].id;
    const data = await getJSON('/api/section/' + encodeURIComponent(book.id));
    if (state.panel !== 'book') return;
    const options = book.toc.map(s => `<optgroup label="${escapeHTML(SUBJECTS[s.subject] || s.subject)}">${s.sections.map(x =>
      `<option value="${x.id}"${x.id === book.id ? ' selected' : ''}>${escapeHTML(x.id + ' · ' + x.title)}</option>`).join('')}</optgroup>`).join('');
    const drawn = renderSection(data);
    const legend = !book.answer ? '' : drawn.first.length
      ? '<div class="book-legend"><span><i class="lq"></i>서버가 질문에 가장 가깝다고 본 소단원 · 답변은 이 섹션 전체를 인용했어요</span></div>'
      : drawn.hit ? '<div class="book-legend"><span><i class="lq"></i>답변에 인용된 문장</span></div>' : '';
    body.innerHTML = `<select class="book-pick" aria-label="교재 섹션 고르기">${options}</select>${legend}
      <div class="book-paper"><div class="panel-label">${escapeHTML(SUBJECTS[data.subject] || data.subject)} · ${escapeHTML(data.id)}</div><h3>${escapeHTML(data.title)}</h3>${drawn.html}</div>`;
    body.querySelector('.book-pick').addEventListener('change', event => {
      book.id = event.target.value; book.answer = ''; book.question = ''; drawBook();
    });
    drawn.first.forEach(k => { const el = body.querySelector(`[data-b="${k}"]`); if (el) el.classList.add('quoted', 'focus'); });
    const target = drawn.first.length ? body.querySelector(`[data-b="${drawn.first[0]}"]`) : body.querySelector('mark, .quoted');
    body.scrollTop = target ? Math.max(0, target.offsetTop - body.offsetTop - 90) : 0;
  }

  renderBook = function () { drawBook(); };
  function showBook(id, answer, question) {
    book.id = id; book.answer = answer || ''; book.question = question || '';
    if (state.panel === 'book') drawBook(); else openPanel('book');
  }
  const scriptedOpenPanel = openPanel;
  openPanel = function (type) {
    if (type === 'book' && !$('messages').children.length) { ensureChat(); addAI('<p>왼쪽에서 교재를 골라 읽어 보세요. 궁금한 것은 여기에 물어보면 돼요.</p>'); }
    scriptedOpenPanel(type);
    if (type === 'book') $('panel-title').innerHTML = icon('book') + '교재 함께 보기';
  };

  async function citation(body, id, answer, question) {
    const card = document.createElement('button');
    card.className = 'ref-card';
    card.type = 'button';
    card.innerHTML = `${icon('book')}<span>${escapeHTML(id)}<small>교재에서 근거 보기</small></span>${icon('chevron', true)}`;
    card.addEventListener('click', event => { event.stopPropagation(); showBook(id, answer, question); });
    body.appendChild(card);
    try {
      const data = await getJSON('/api/section/' + encodeURIComponent(id));
      if (data.title) card.querySelector('span').innerHTML = `${escapeHTML(SUBJECTS[data.subject] || data.subject)} · ${escapeHTML(data.title)}<small>${escapeHTML(id)} · 교재에서 근거 보기</small>`;
    } catch (e) { /* 제목을 못 가져오면 ID만 보여 준다 */ }
  }

  async function ask(text) {
    busy = true;
    const messages = [{ role: 'user', content: text }];
    const el = addAI('<p class="live-wait">교재를 찾아보는 중이에요…</p>');
    const body = el.querySelector('.message-body');
    const started = performance.now();
    try {
      const response = await fetch('/api/tutor', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec_version: '1.0', session_id: 's_demo_' + Date.now(), messages }),
      });
      if (!response.ok) throw new Error('status ' + response.status);
      const data = await response.json();
      body.innerHTML = `<span class="live-action${tone(data.action)}">${escapeHTML(ACTIONS[data.action] || data.action)}</span><div class="live-answer">${render(data.answer)}</div>`;
      body.dataset.seconds = ((performance.now() - started) / 1000).toFixed(1);
      const answer = body.querySelector('.live-answer');
      if (answer.scrollHeight > 190) {
        answer.classList.add('folded');
        const more = document.createElement('button');
        more.type = 'button';
        more.className = 'live-more';
        more.textContent = '펼쳐 보기';
        more.addEventListener('click', event => {
          event.stopPropagation();
          more.textContent = answer.classList.toggle('folded') ? '펼쳐 보기' : '접기';
        });
        answer.after(more);
      }
      const cited = data.citations || [];
      for (const id of cited) citation(body, id, data.answer, text);
      // 교재는 칠할 곳이 있을 때만 저절로 띄운다. 없으면 닫아 두고, 인용 카드를 누르면 열린다.
      let painted = false;
      if (cited.length) {
        try {
          const section = await getJSON('/api/section/' + encodeURIComponent(cited[0]));
          const keep = book.answer;
          book.answer = data.answer;
          const drawn = renderSection(section);
          book.answer = keep;
          painted = drawn.first.length > 0 || drawn.hit > 0;
        } catch (e) { painted = false; }
      }
      if (painted) showBook(cited[0], data.answer, text);
      else if (state.panel === 'book') closePanel(false);
    } catch (e) {
      body.innerHTML = '<p>서버에서 답을 받지 못했어요. 대회 서버가 켜져 있는지 확인해 주세요.</p>';
    }
    busy = false;
    scrollChat();
  }

  submitText = function (text) {
    const t = text.trim();
    if (!t) return;
    if (busy) { toast('앞의 답변을 기다리는 중이에요.'); return; }
    if (demo.active) stopDemo();
    ensureChat();
    addUser(t);
    ask(t);
  };

  const scriptedNewChat = newChat;
  newChat = function () {
    book.answer = ''; book.question = '';
    return scriptedNewChat.apply(this, arguments);
  };

  const lead = document.querySelector('.welcome p');
  if (lead) lead.textContent = '교재에서 근거를 찾아 함께 살펴봐요.';
})();
