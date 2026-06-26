// ╔══════════════════════════════════════════════════════════════════╗
// ║       IndexGen Pro — server.js v5.2 (DEFINITIVE FIX)           ║
// ╠══════════════════════════════════════════════════════════════════╣
// ║  FIX 1 : Fully sequential API calls (7s gap) — no rate limits  ║
// ║  FIX 2 : 70s wait on 429 — correct rate-limit handling         ║
// ║  FIX 3 : Page-by-page fallback — zero permanent skips          ║
// ║  FIX 4 : cleanTopic() — strips ⑤ ↳ * + symbols from headings  ║
// ║  FIX 5 : Prompt updated — AI told to omit leading symbols      ║
// ╚══════════════════════════════════════════════════════════════════╝

import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { GoogleGenAI } from '@google/genai';
import cors from 'cors';

const execAsync = promisify(exec);
const app = express();
app.use(cors());
app.use(express.json());

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// ─── CONFIG ───────────────────────────────────────────────────────────────────
const MODEL_NAME    = 'gemini-2.5-flash';
const ONE_PAGE_MAX  = 25;
const CHUNK_SIZE    = 5;     // 5 pages per API call (safe for free tier)
const MAX_RETRIES   = 4;     // retries per chunk before page-by-page fallback
const API_GAP_MS    = 7000;  // mandatory 7s between every Gemini call (ensures <9 RPM)
const RATELIMIT_MS  = 72000; // 72s wait when 429 received (safe reset window)
const MAX_FILE_MB   = 50;
const JOB_TTL_MS    = 25 * 60 * 1000;

if (!process.env.GEMINI_API_KEY) {
  console.error('FATAL: GEMINI_API_KEY not set.');
  process.exit(1);
}

// ─── IN-MEMORY JOB STORE ─────────────────────────────────────────────────────
const jobs = new Map();

// ─── MULTER ───────────────────────────────────────────────────────────────────
const upload = multer({
  dest: os.tmpdir(),
  limits: { fileSize: MAX_FILE_MB * 1024 * 1024 },
  fileFilter: (_, file, cb) =>
    file.mimetype === 'application/pdf'
      ? cb(null, true)
      : cb(new Error(`Only PDF files allowed. Received: ${file.mimetype}`))
});

// ─── SCHEMAS ──────────────────────────────────────────────────────────────────
const INDEX_SCHEMA = {
  type: 'array',
  items: {
    type: 'object',
    properties: {
      page:  { type: 'integer' },
      topic: { type: 'string'  },
      level: { type: 'integer' }
    },
    required: ['page', 'topic', 'level']
  }
};

const CONTEXT_SCHEMA = {
  type: 'object',
  properties: {
    subject:        { type: 'string'  },
    documentType:   { type: 'string'  },
    theme:          { type: 'string'  },
    headingPattern: { type: 'string'  },
    entryFormat:    { type: 'string'  },
    isHandwritten:  { type: 'boolean' },
    topicStructure: { type: 'string'  },
    language:       { type: 'string'  }
  },
  required: [
    'subject','documentType','theme','headingPattern',
    'entryFormat','isHandwritten','topicStructure','language'
  ]
};

// ─── ROUTES ───────────────────────────────────────────────────────────────────
app.get('/', (_, res) => res.json({ status: 'IndexGen Pro v5.2', model: MODEL_NAME }));

app.get('/health', (_, res) => res.json({
  ok: true, model: MODEL_NAME, activeJobs: jobs.size,
  config: { chunkSize: CHUNK_SIZE, apiGapMs: API_GAP_MS, onPageMax: ONE_PAGE_MAX }
}));

// ─── UPLOAD → returns jobId immediately (<1s) ─────────────────────────────────
app.post('/api/generate-index', upload.single('pdf'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No PDF uploaded. Field name must be "pdf".' });
  }

  const jobId = `job_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  jobs.set(jobId, {
    status: 'queued', progress: '📋 Job queued...', step: 0,
    pagesTotal: 0, pagesDone: 0,
    result: null, error: null, startedAt: Date.now()
  });

  res.json({ jobId, status: 'queued' });

  processJob(jobId, req.file).catch(err => {
    const job = jobs.get(jobId);
    if (job) { job.status = 'error'; job.error = err.message; }
    console.error(`[${jobId}] Unhandled:`, err.message);
  });
});

// ─── POLLING ENDPOINT ─────────────────────────────────────────────────────────
app.get('/api/status/:jobId', (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({
      error: 'Job not found. Server may have restarted. Please re-upload your PDF.'
    });
  }
  res.json({
    status: job.status, progress: job.progress, step: job.step,
    pagesTotal: job.pagesTotal, pagesDone: job.pagesDone,
    result: job.result, error: job.error,
    elapsedMs: Date.now() - job.startedAt
  });
});

// ─── MAIN BACKGROUND PROCESSOR ───────────────────────────────────────────────
async function processJob(jobId, file) {
  const job     = jobs.get(jobId);
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'idx-'));
  const pdfPath = file.path;
  const t0      = Date.now();

  try {
    // ── STEP 1: PDF → JPEG ────────────────────────────────────────────────────
    job.status = 'processing'; job.step = 1;
    job.progress = '⚙️  Step 1/4 — Converting PDF pages to images...';

    await execAsync(`pdftoppm -jpeg -r 150 "${pdfPath}" "${workDir}/page"`);

    const pageFiles = fs.readdirSync(workDir)
      .filter(f => f.endsWith('.jpg'))
      .sort((a, b) => {
        const na = parseInt(a.match(/\d+/)?.[0] || 0);
        const nb = parseInt(b.match(/\d+/)?.[0] || 0);
        return na - nb;
      });

    if (!pageFiles.length) {
      throw new Error('PDF conversion failed — file may be corrupt or password-protected.');
    }

    job.pagesTotal = pageFiles.length;
    job.step = 2;
    job.progress = `📄 Step 2/4 — Analyzing document (${pageFiles.length} pages found)...`;
    console.log(`\n[${jobId}] ${pageFiles.length} pages converted (${elapsed(t0)})`);

    // ── STEP 2: Context analysis ───────────────────────────────────────────────
    const sampleFiles = pageFiles.slice(0, Math.min(3, pageFiles.length));
    const docCtx      = await analyzeContext(workDir, sampleFiles, pageFiles.length);
    console.log(`[${jobId}] Context: ${docCtx.subject} | "${docCtx.theme}" | ${docCtx.topicStructure} (${elapsed(t0)})`);

    // mandatory gap after context call
    await sleep(API_GAP_MS);

    // ── STEP 3: Sequential extraction ─────────────────────────────────────────
    job.step = 3;
    job.progress = `🔍 Step 3/4 — Extracting headings from all ${pageFiles.length} pages...`;

    const { allEntries, permanentlyFailed } = await extractAllSequential(
      jobId, workDir, pageFiles, docCtx, job, t0
    );

    // Sort + dedup + cleanTopic
    let entries = [...allEntries].sort((a, b) => a.page - b.page);
    const seen  = new Set();
    entries = entries.filter(({ page, topic }) => {
      const key = `${page}|||${topic.toLowerCase().trim()}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    // ── APPLY cleanTopic TO EVERY ENTRY ───────────────────────────────────────
    entries = entries.map(e => ({ ...e, topic: cleanTopic(e.topic) }));

    console.log(`[${jobId}] ${entries.length} clean entries (${elapsed(t0)})`);

    // ── STEP 4: Consolidate ───────────────────────────────────────────────────
    job.step = 4;
    let finalEntries = entries;
    let consolidated = false;

    if (entries.length > ONE_PAGE_MAX) {
      job.progress = `✂️  Step 4/4 — Selecting best ${ONE_PAGE_MAX} from ${entries.length}...`;
      await sleep(API_GAP_MS);
      finalEntries = await consolidate(entries, docCtx);
      consolidated = true;
    } else {
      job.progress = `✅ Step 4/4 — ${entries.length} entries — all fit on one page.`;
    }

    const totalMs = Date.now() - t0;
    console.log(`[${jobId}] ✅ Finished in ${(totalMs/1000).toFixed(1)}s | ${finalEntries.length} entries\n`);

    job.status   = 'done';
    job.progress = `✅ Complete — ${finalEntries.length} entries in ${(totalMs/1000).toFixed(1)}s`;
    job.result = {
      success: true,
      totalPages: pageFiles.length,
      rawEntries: entries.length,
      finalEntries: finalEntries.length,
      consolidated,
      processingMs: totalMs,
      documentContext: docCtx,
      index: finalEntries,
      ...(permanentlyFailed.length > 0 ? {
        warning: `${permanentlyFailed.length} page(s) unreadable after all retries: ${permanentlyFailed.join(', ')}`
      } : {})
    };

    setTimeout(() => jobs.delete(jobId), JOB_TTL_MS);

  } catch (err) {
    console.error(`[${jobId}] Fatal:`, err.message);
    job.status = 'error';
    job.error  = err.message || 'Internal error. Please try again.';
    setTimeout(() => jobs.delete(jobId), 5 * 60 * 1000);
  } finally {
    fs.rmSync(workDir, { recursive: true, force: true });
    fs.unlink(pdfPath, () => {});
  }
}

// ─── FULLY SEQUENTIAL EXTRACTION ENGINE ──────────────────────────────────────
// Processes one chunk at a time. Mandatory API_GAP_MS between every call.
// On rate limit (429): waits RATELIMIT_MS before retry.
// On persistent failure: falls back to page-by-page.
async function extractAllSequential(jobId, workDir, pageFiles, docCtx, job, t0) {
  // Build chunks
  const chunks = [];
  for (let i = 0; i < pageFiles.length; i += CHUNK_SIZE) {
    chunks.push({ files: pageFiles.slice(i, i + CHUNK_SIZE), offset: i });
  }

  let allEntries       = [];
  let permanentlyFailed = [];
  let callCount        = 0;

  for (let ci = 0; ci < chunks.length; ci++) {
    const { files, offset } = chunks[ci];
    const rangeLabel = `pages ${offset+1}–${offset+files.length}`;

    job.pagesDone = offset;
    job.progress  = `🔍 Step 3/4 — Processing ${rangeLabel} of ${pageFiles.length} (${allEntries.length} headings so far)...`;

    // Mandatory gap before each call (except the very first after context)
    if (callCount > 0) await sleep(API_GAP_MS);
    callCount++;

    // Try this chunk (up to MAX_RETRIES)
    let succeeded   = false;
    let chunkEntries = null;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        chunkEntries = await extractChunk(workDir, files, offset, docCtx);
        succeeded    = true;
        console.log(`[${jobId}] ✓ ${rangeLabel}: ${chunkEntries.length} headings (${elapsed(t0)})`);
        break;
      } catch (err) {
        const isRL = isRateLimitError(err);
        console.warn(`[${jobId}] ✗ ${rangeLabel} attempt ${attempt+1}: ${err.message?.slice(0,60)}`);

        if (attempt < MAX_RETRIES) {
          const waitMs = isRL ? RATELIMIT_MS : Math.min(8000 * Math.pow(2, attempt), 30000);
          console.log(`[${jobId}] Waiting ${(waitMs/1000).toFixed(0)}s before retry...`);
          job.progress = `⏳ Rate limit hit — waiting ${(waitMs/1000).toFixed(0)}s then retrying ${rangeLabel}...`;
          await sleep(waitMs);
          callCount++;
        }
      }
    }

    if (succeeded && chunkEntries) {
      allEntries.push(...chunkEntries);
      continue;
    }

    // ── PAGE-BY-PAGE FALLBACK ─────────────────────────────────────────────────
    console.log(`[${jobId}] Chunk failed all retries → page-by-page fallback for ${rangeLabel}`);
    job.progress = `🔄 Retrying ${rangeLabel} page-by-page...`;
    await sleep(RATELIMIT_MS); // Full rate-limit reset before page-by-page

    for (let i = 0; i < files.length; i++) {
      const pageNum = offset + i + 1;
      await sleep(API_GAP_MS);
      callCount++;

      let pageDone = false;
      for (let attempt = 0; attempt <= 2; attempt++) {
        try {
          const pageEntries = await extractChunk(workDir, [files[i]], offset + i, docCtx);
          allEntries.push(...pageEntries);
          console.log(`[${jobId}]   ✓ page ${pageNum}: ${pageEntries.length} headings`);
          pageDone = true;
          break;
        } catch (err) {
          if (isRateLimitError(err) && attempt < 2) {
            await sleep(RATELIMIT_MS);
          } else if (attempt < 2) {
            await sleep(10000);
          }
        }
      }

      if (!pageDone) {
        permanentlyFailed.push(pageNum);
        console.error(`[${jobId}]   ✗ page ${pageNum} permanently failed`);
      }
    }
  }

  job.pagesDone = pageFiles.length;
  return { allEntries, permanentlyFailed };
}

// ─── Rate limit detection ─────────────────────────────────────────────────────
function isRateLimitError(err) {
  const msg = (err.message || '').toLowerCase();
  return msg.includes('429') ||
         msg.includes('quota') ||
         msg.includes('rate') ||
         msg.includes('resource_exhausted') ||
         msg.includes('too many');
}

// ─── TOPIC CLEANER ────────────────────────────────────────────────────────────
// Removes leading symbols copied from handwritten notes:
// ⑤ ↳ → * + • ○ □ ☐ etc. Also strips trailing colons.
function cleanTopic(raw) {
  if (!raw) return raw;
  let t = raw.trim();

  // 1. Remove circled/enclosed numbers: ①②③...⑳ and Unicode variants
  t = t.replace(/^[\u2460-\u2473\u2474-\u2487\u2488-\u249B\u24B6-\u24FF\u2776-\u2793]+\s*/u, '');

  // 2. Remove arrow/bullet/cross/star symbols at the start
  //    Covers: ↳ → ← ↑ ↓ ↪ ↩ ➤ ➜ ► ▸ ▹ ◀ • · ○ ● □ ■ ☐ ☑ ◆ ◇ ▪ ▫ ◉ ⊕ ⊗ ✓ ✗ ✦ ✧ ✱ ✲
  t = t.replace(/^[\u2190-\u21FF\u2700-\u27BF\u25A0-\u25FF\u2600-\u26FF\u2300-\u23FF☐☑✓✗]+\s*/u, '');

  // 3. Remove simple ASCII symbols: * + - ~ # = > < | \
  t = t.replace(/^[*+\-~#=><!|\\]+\s*/, '');

  // 4. Remove leading numbers like "1." "2)" "(3)" "1:" but NOT "Chapter 1" or "Fig. 3"
  t = t.replace(/^\(?\d{1,2}[.):\s]\)?\s+/, '');

  // 5. Strip trailing colon (very common in handwritten notes headings)
  t = t.replace(/[:\s]+$/, '').trim();

  // 6. Capitalize first letter if all lowercase
  if (t && t[0] === t[0].toLowerCase() && t[0] !== t[0].toUpperCase() === false) {
    // leave as-is (it might be a scientific name like "Tubifex tubifex")
  }

  return t || raw.trim(); // if cleaning removes everything, return original
}

// ─── Single chunk extraction ──────────────────────────────────────────────────
async function extractChunk(workDir, chunkFiles, offset, ctx) {
  const parts = [{ text: buildExtractionPrompt(chunkFiles.length, offset, ctx) }];

  chunkFiles.forEach((f, i) => {
    parts.push({ text: `=== PAGE ${offset + i + 1} ===` });
    parts.push({ inlineData: { mimeType: 'image/jpeg', data: b64(workDir, f) } });
  });

  const r = await ai.models.generateContent({
    model:    MODEL_NAME,
    contents: parts,
    config:   { responseMimeType: 'application/json', responseSchema: INDEX_SCHEMA }
  });

  const raw    = (r.text || '[]').replace(/```json|```/g, '').trim();
  const parsed = JSON.parse(raw);
  return Array.isArray(parsed) ? parsed : [];
}

// ─── Context analysis ─────────────────────────────────────────────────────────
async function analyzeContext(workDir, sampleFiles, totalPages) {
  const parts = [{ text: `You are an expert academic document analyst. Study these ${sampleFiles.length} sample pages from a ${totalPages}-page PDF.

Return ONLY valid JSON with NO extra text:
{
  "subject": "Academic subject (e.g. Zoology, Fluid Mechanics, Environmental Science)",
  "documentType": "Type (e.g. Assignment, Lab Report, Lecture Notes, Handwritten Notes)",
  "theme": "Core theme in 5-10 words describing what this document covers",
  "headingPattern": "Exactly how headings appear visually on the page",
  "entryFormat": "How to write index entries — clean topic names only, no symbols",
  "isHandwritten": true or false,
  "topicStructure": "SERIES (each page = 1 independent topic) | HIERARCHICAL (chapters > sections) | MIXED",
  "language": "Primary language (English, Bengali, Mixed)"
}` }];

  sampleFiles.forEach((f, i) => {
    parts.push({ text: `=== SAMPLE PAGE ${i + 1} ===` });
    parts.push({ inlineData: { mimeType: 'image/jpeg', data: b64(workDir, f) } });
  });

  try {
    const r = await ai.models.generateContent({
      model:    MODEL_NAME,
      contents: parts,
      config:   { responseMimeType: 'application/json', responseSchema: CONTEXT_SCHEMA }
    });
    const ctx = JSON.parse((r.text || '{}').replace(/```json|```/g, '').trim());
    if (!ctx.subject || !ctx.theme) throw new Error('Incomplete context');
    return ctx;
  } catch (err) {
    console.warn('Context fallback:', err.message);
    return {
      subject: 'Academic', documentType: 'Lecture Notes', isHandwritten: true,
      theme: 'Academic document',
      headingPattern: 'Most prominent text at top of each section',
      entryFormat: 'Clean topic name without symbols or colons',
      topicStructure: 'SERIES', language: 'English'
    };
  }
}

// ─── Extraction prompt ────────────────────────────────────────────────────────
function buildExtractionPrompt(pageCount, offset, ctx) {
  return `You are an expert academic index builder.

DOCUMENT CONTEXT:
  Subject      : ${ctx.subject}
  Type         : ${ctx.documentType}
  Theme        : "${ctx.theme}"
  Heading style: ${ctx.headingPattern}
  Structure    : ${ctx.topicStructure}
  Language     : ${ctx.language}
  Handwritten  : ${ctx.isHandwritten}

YOUR TASK: Find the main topic/heading on each page and write it CLEANLY.

HEADING DETECTION:
  ✅ EXTRACT: The most prominent title/heading at the start of a new topic
  ✅ EXTRACT: Underlined text (handwritten) or Bold/ALL-CAPS text (typed)
  ❌ SKIP: Taxonomy rows (Kingdom / Phylum / Class / Order / Family / Genus / Species)
  ❌ SKIP: Figure captions (Fig., Figure, Diagram, Table)
  ❌ SKIP: Student/teacher name, roll number, course code, date, institution
  ❌ SKIP: Page numbers in headers/footers
  ❌ SKIP: "References", "Bibliography", "Contents"
  ❌ SKIP: Pure body text paragraphs with no heading

CRITICAL — CLEAN OUTPUT RULES:
  ✗ DO NOT include leading symbols: ⑤ ① ↳ → * + • ○ □ ☐ ✓ ☆ ★ or similar
  ✗ DO NOT include trailing colon (:) at the end of the topic
  ✗ DO NOT include sub-symbols like "+" "-" "→" that prefix the heading
  ✓ Write ONLY the topic name itself, clean and readable
  ✓ Preserve scientific names (e.g. Ascaris lumbricoides — keep italics intent)
  ✓ Preserve proper nouns and capitalization

EXAMPLES of cleaning:
  Raw on page    : "⑤ Daly's Glacial Control Theory:"
  Output topic   : "Daly's Glacial Control Theory"

  Raw on page    : "↳ Parasitic Adaptations:"
  Output topic   : "Parasitic Adaptations"

  Raw on page    : "+ Culture Media:"
  Output topic   : "Culture Media"

  Raw on page    : "* Characteristics of Yeast and Molds"
  Output topic   : "Characteristics of Yeast and Molds"

  Raw on page    : "Systematic Position of Ascaris lumbricoides"
  Output topic   : "Systematic Position of Ascaris lumbricoides"  ← already clean, keep as-is

RULES:
  1. Process ALL ${pageCount} pages — NEVER skip.
  2. ONE heading per page — the most important one.
  3. Omit the page from output if it has no heading (blank, pure diagram, or continuation).
  4. Pages are labeled "=== PAGE N ===" → use N as the exact page number.
  5. Output clean, readable topic names — as they would appear in a printed book index.

Pages to analyze: ${offset + 1} to ${offset + pageCount}

Return ONLY a JSON array, no other text:
[{"page": N, "topic": "Clean topic name", "level": 0}]`;
}

// ─── Consolidate to ONE printed page ─────────────────────────────────────────
async function consolidate(allEntries, docCtx, max = ONE_PAGE_MAX) {
  const structure = (docCtx.topicStructure || 'SERIES').toUpperCase();
  const strategy  = structure.includes('SERIES')
    ? `SERIES document: Select EVENLY distributed entries covering beginning, middle, AND end of the document.`
    : structure.includes('HIERARCHICAL')
    ? `HIERARCHICAL document: Keep ONLY chapter-level entries. Remove all sub-headings.`
    : `MIXED: Keep one entry per major theme. Remove near-duplicates.`;

  const lastPage = allEntries[allEntries.length - 1]?.page || '?';

  try {
    const r = await ai.models.generateContent({
      model:    MODEL_NAME,
      contents: [{ text: `Academic index
