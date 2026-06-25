// ╔══════════════════════════════════════════════════════════════════╗
// ║         IndexGen Pro — server.js v5.0 (POLLING + QUALITY)      ║
// ╠══════════════════════════════════════════════════════════════════╣
// ║  FIX 1: POST returns jobId in <1s (no more 120s timeout)        ║
// ║  FIX 2: Parallel chunk processing (3 concurrent Gemini calls)   ║
// ║  FIX 3: Better prompt → real contextual headlines from any PDF  ║
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
const MODEL_NAME   = 'gemini-2.5-flash';
const ONE_PAGE_MAX = 25;
const CHUNK_SIZE   = 10;
const CONCURRENT   = 3;
const MAX_RETRIES  = 3;
const MAX_FILE_MB  = 50;
const JOB_TTL_MS   = 20 * 60 * 1000; // 20 min auto-cleanup

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
    'subject','documentType','theme',
    'headingPattern','entryFormat',
    'isHandwritten','topicStructure','language'
  ]
};

// ─── ROUTES ───────────────────────────────────────────────────────────────────
app.get('/', (_, res) => res.json({ status: 'IndexGen Pro v5', model: MODEL_NAME }));

app.get('/health', (_, res) => res.json({
  ok: true,
  model: MODEL_NAME,
  activeJobs: jobs.size,
  keySet: !!process.env.GEMINI_API_KEY,
  config: { chunkSize: CHUNK_SIZE, concurrent: CONCURRENT, onPageMax: ONE_PAGE_MAX }
}));

// ─── UPLOAD → returns jobId IMMEDIATELY (<1s) ─────────────────────────────────
app.post('/api/generate-index', upload.single('pdf'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No PDF uploaded. Field name must be "pdf".' });
  }

  const jobId = `job_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

  jobs.set(jobId, {
    status:     'queued',
    progress:   'Job queued — server waking up...',
    step:       0,
    pagesTotal: 0,
    pagesDone:  0,
    result:     null,
    error:      null,
    startedAt:  Date.now()
  });

  // ✅ Respond in <1s — no timeout possible
  res.json({ jobId, status: 'queued' });

  // 🔄 Process in background
  processJob(jobId, req.file).catch(err => {
    const job = jobs.get(jobId);
    if (job) { job.status = 'error'; job.error = err.message || 'Unknown error'; }
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
    status:     job.status,
    progress:   job.progress,
    step:       job.step,
    pagesTotal: job.pagesTotal,
    pagesDone:  job.pagesDone,
    result:     job.result,
    error:      job.error,
    elapsedMs:  Date.now() - job.startedAt
  });
});

// ─── BACKGROUND PROCESSOR ─────────────────────────────────────────────────────
async function processJob(jobId, file) {
  const job     = jobs.get(jobId);
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'idx-'));
  const pdfPath = file.path;
  const t0      = Date.now();

  try {
    // ── STEP 1: PDF → JPEG ────────────────────────────────────────────────────
    job.status   = 'processing';
    job.step     = 1;
    job.progress = '⚙️  Step 1/4 — Converting PDF pages to images...';

    await execAsync(`pdftoppm -jpeg -r 150 "${pdfPath}" "${workDir}/page"`);

    const pageFiles = fs.readdirSync(workDir)
      .filter(f => f.endsWith('.jpg'))
      .sort((a, b) => {
        // Natural sort: page-1.jpg, page-2.jpg, ..., page-10.jpg
        const numA = parseInt(a.match(/\d+/)?.[0] || 0);
        const numB = parseInt(b.match(/\d+/)?.[0] || 0);
        return numA - numB;
      });

    if (!pageFiles.length) {
      throw new Error('PDF conversion failed — file may be corrupt or password-protected.');
    }

    job.pagesTotal = pageFiles.length;
    job.progress   = `📄 Step 2/4 — Analyzing document type & structure (${pageFiles.length} pages)...`;
    job.step       = 2;
    console.log(`\n[${jobId}] ${pageFiles.length} pages converted (${elapsed(t0)})`);

    // ── STEP 2: Context analysis ───────────────────────────────────────────────
    // Use first 3 pages for better context (cover + first content pages)
    const sampleFiles = pageFiles.slice(0, Math.min(3, pageFiles.length));
    const docCtx      = await analyzeContext(workDir, sampleFiles, pageFiles.length);
    console.log(`[${jobId}] Context: ${docCtx.subject} | "${docCtx.theme}" | ${docCtx.topicStructure} (${elapsed(t0)})`);

    // ── STEP 3: Parallel chunk extraction ─────────────────────────────────────
    job.step     = 3;
    job.progress = `🔍 Step 3/4 — Extracting headings from "${docCtx.theme}"...`;

    const chunkTasks = [];
    for (let i = 0; i < pageFiles.length; i += CHUNK_SIZE) {
      chunkTasks.push({ chunk: pageFiles.slice(i, i + CHUNK_SIZE), offset: i });
    }

    let allEntries     = [];
    const failed       = [];
    let processedPages = 0;

    for (let b = 0; b < chunkTasks.length; b += CONCURRENT) {
      const batch   = chunkTasks.slice(b, b + CONCURRENT);
      const settled = await Promise.allSettled(
        batch.map(({ chunk, offset }) => extractWithRetry(workDir, chunk, offset, docCtx))
      );

      settled.forEach((result, idx) => {
        const task  = batch[idx];
        const label = `pages ${task.offset + 1}–${task.offset + task.chunk.length}`;
        if (result.status === 'fulfilled') {
          allEntries.push(...result.value);
          console.log(`[${jobId}] ✓ ${label}: ${result.value.length} headings`);
        } else {
          console.error(`[${jobId}] ✗ ${label}: ${result.reason?.message}`);
          failed.push({ from: task.offset + 1, to: task.offset + task.chunk.length });
        }
        processedPages += task.chunk.length;
      });

      job.pagesDone = Math.min(processedPages, pageFiles.length);
      job.progress  = `🔍 Step 3/4 — Scanned ${job.pagesDone}/${pageFiles.length} pages... (${allEntries.length} headings found)`;
    }

    // Sort + dedup + sanitize
    allEntries.sort((a, b) => a.page - b.page);
    const seen = new Set();
    allEntries = allEntries.filter(({ page, topic }) => {
      const key = `${page}|||${topic.toLowerCase().trim()}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    allEntries = allEntries.map(e => ({
      ...e,
      topic: e.topic.replace(/<(?!\/?em>)[^>]+>/gi, '').trim()
    }));

    console.log(`[${jobId}] ${allEntries.length} unique entries (${elapsed(t0)})`);

    // ── STEP 4: 1-page guarantee ───────────────────────────────────────────────
    job.step = 4;
    let finalEntries = allEntries;
    let consolidated = false;

    if (allEntries.length > ONE_PAGE_MAX) {
      job.progress = `✂️  Step 4/4 — Selecting best ${ONE_PAGE_MAX} from ${allEntries.length} headings...`;
      finalEntries = await consolidate(allEntries, docCtx);
      consolidated = true;
    } else {
      job.progress = `✅ Step 4/4 — ${allEntries.length} entries fit on one page. Done!`;
    }

    const totalMs = Date.now() - t0;
    console.log(`[${jobId}] ✅ Finished in ${(totalMs / 1000).toFixed(1)}s | ${finalEntries.length} entries\n`);

    job.status   = 'done';
    job.progress = `✅ Complete — ${finalEntries.length} index entries in ${(totalMs / 1000).toFixed(1)}s`;
    job.result   = {
      success:         true,
      totalPages:      pageFiles.length,
      rawEntries:      allEntries.length,
      finalEntries:    finalEntries.length,
      consolidated,
      processingMs:    totalMs,
      documentContext: docCtx,
      index:           finalEntries,
      ...(failed.length ? {
        warning: `Some pages skipped due to API errors: ${failed.map(c => `${c.from}–${c.to}`).join(', ')}`
      } : {})
    };

    // Auto-cleanup
    setTimeout(() => { jobs.delete(jobId); console.log(`[${jobId}] Cleaned up.`); }, JOB_TTL_MS);

  } catch (err) {
    console.error(`[${jobId}] Fatal:`, err.message);
    job.status   = 'error';
    job.progress = 'Processing failed.';
    job.error    = err.message || 'Internal server error. Please try again.';
    setTimeout(() => jobs.delete(jobId), 5 * 60 * 1000);
  } finally {
    fs.rmSync(workDir, { recursive: true, force: true });
    fs.unlink(pdfPath, () => {});
  }
}

// ─── Phase 1: Document context analysis ──────────────────────────────────────
async function analyzeContext(workDir, sampleFiles, totalPages) {
  const parts = [{
    text: `You are an expert academic document analyst. Study these ${sampleFiles.length} sample pages from a ${totalPages}-page PDF.

Return ONLY valid JSON with NO extra text or markdown:
{
  "subject": "The academic subject (e.g. Zoology, Fluid Mechanics, Environmental Science, Chemistry)",
  "documentType": "Type of document (e.g. Assignment, Lab Report, Research Paper, Lecture Notes, Textbook Chapter)",
  "theme": "Core theme in 5-10 words (e.g. Systematic classification of freshwater invertebrates)",
  "headingPattern": "Describe exactly how headings look (e.g. Bold centered text like 'Systematic Position of [Species Name]' at top of each page)",
  "entryFormat": "How to format index entries (e.g. 'Systematic Position of Dero dorsalis' — keep Latin names in italic, preserve capitalization)",
  "isHandwritten": false,
  "topicStructure": "SERIES (each page = 1 topic, all equal importance) OR HIERARCHICAL (chapters > sub-sections) OR MIXED",
  "language": "Primary language (e.g. English, Bengali, Mixed)"
}`
  }];

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
    // Validate required fields
    if (!ctx.subject || !ctx.theme) throw new Error('Incomplete context');
    return ctx;
  } catch (err) {
    console.warn('Context analysis fallback:', err.message);
    return {
      subject: 'Academic', documentType: 'Assignment',
      isHandwritten: false,
      theme: 'Academic document',
      headingPattern: 'Most prominent bold or underlined text at top of each new section or topic',
      entryFormat: 'Main topic title exactly as written on the page',
      topicStructure: 'SERIES',
      language: 'English'
    };
  }
}

// ─── Phase 2: Extract headings with retry ────────────────────────────────────
async function extractWithRetry(workDir, chunk, offset, docCtx) {
  let lastErr;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await extractChunk(workDir, chunk, offset, docCtx);
    } catch (err) {
      lastErr = err;
      // Don't retry on permanent errors
      if (err.message?.includes('SAFETY') || err.message?.includes('400')) break;
      if (attempt < MAX_RETRIES) {
        const delay = 1500 * Math.pow(2, attempt); // exponential backoff
        await sleep(delay);
      }
    }
  }
  throw lastErr;
}

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

function buildExtractionPrompt(pageCount, offset, ctx) {
  const isHandwritten = ctx.isHandwritten;

  const headingRules = isHandwritten ? `
HANDWRITTEN DOCUMENT RULES:
  ✅ EXTRACT: Underlined text at the start of a new topic
  ✅ EXTRACT: Text with special symbols (★ □ ☐) beside it
  ✅ EXTRACT: Larger/bolder handwriting at the top of a new section
  ❌ SKIP: Taxonomy rows (Kingdom / Phylum / Class / Order / Family / Genus / Species)
  ❌ SKIP: Plain body text (not underlined or specially marked)` : `
TYPED/PRINTED DOCUMENT RULES:
  ✅ EXTRACT: Bold standalone line (not part of a sentence)
  ✅ EXTRACT: ALL-CAPS title line
  ✅ EXTRACT: Centered heading at start of a new section
  ✅ EXTRACT: Numbered heading like "Chapter 3: Heat Transfer"
  ❌ SKIP: Sub-headings that are NOT the main topic title
  ❌ SKIP: Numbered list items (1. 2. 3. that are content, not headings)
  ❌ SKIP: Bold text inside a paragraph`;

  return `You are an expert academic index builder. Extract EXACTLY ONE heading per page that would appear in a printed index.

DOCUMENT CONTEXT:
  Subject: ${ctx.subject}
  Type: ${ctx.documentType}
  Theme: "${ctx.theme}"
  How headings look: ${ctx.headingPattern}
  How to format entries: ${ctx.entryFormat}
  Document structure: ${ctx.topicStructure}
  Language: ${ctx.language}

${headingRules}

UNIVERSAL SKIP LIST (NEVER extract these regardless of formatting):
  ✗ Taxonomy rows: Kingdom / Phylum / Class / Order / Family / Genus / Species
  ✗ Figure/diagram captions: "Fig. 3.1", "Figure:", "Diagram:"
  ✗ Page headers/footers (repeated on every page)
  ✗ Student name, roll number, institution name, course code
  ✗ "References", "Bibliography", "Contents", "Table of Contents"
  ✗ Dates like "January 2025"

EXTRACTION RULES:
  1. Process ALL ${pageCount} pages — never skip a page.
  2. Extract the MOST PROMINENT heading/title on each page.
  3. COPY TEXT VERBATIM — exact spelling, exact capitalization, exact punctuation.
  4. If a page has NO heading (pure body text, diagram only, or blank) → DO NOT add an entry for that page.
  5. If a page is clearly a CONTINUATION of the previous topic with no new heading → skip it.
  6. Pages are labeled "=== PAGE N ===" — use N as the exact page number.

OUTPUT: JSON array only, NO other text.
[{"page": N, "topic": "Exact heading text here", "level": 0}]

Pages to process: ${offset + 1} to ${offset + pageCount}`;
}

// ─── Phase 3: Consolidate to ONE printed page ─────────────────────────────────
async function consolidate(allEntries, docCtx, max = ONE_PAGE_MAX) {
  const structure = (docCtx.topicStructure || 'SERIES').toUpperCase();

  const strategyNote = structure.includes('SERIES')
    ? `This is a SERIES document (each page = one independent topic). 
       Select entries EVENLY distributed across the full page range.
       Cover: beginning, middle sections, AND end. Do NOT cluster entries at the start.`
    : structure.includes('HIERARCHICAL')
    ? `This is a HIERARCHICAL document (chapters contain sub-sections).
       Keep ONLY chapter-level entries. Remove all sub-headings.
       Result should be a clean chapter list.`
    : `This is a MIXED document. 
       Keep one entry per major theme. Remove similar/duplicate topics.`;

  const prompt = `You are an academic index editor. A printed index page fits maximum ${max} entries.

Document: "${docCtx.theme}" (${docCtx.subject} ${docCtx.documentType})
Total entries to choose from: ${allEntries.length}
Target: exactly ${max} entries

SELECTION STRATEGY:
${strategyNote}

STRICT RULES:
1. Output EXACTLY ${max} entries — not more, not fewer.
2. Keep original page numbers and topic text COMPLETELY UNCHANGED.
3. Sort by page number (ascending).
4. Ensure good COVERAGE of the full document (from page 1 to page ${allEntries[allEntries.length - 1]?.page || '?'}).
5. If two entries cover the same topic, keep the one with the lower page number.

AVAILABLE ENTRIES:
${JSON.stringify(allEntries)}

Return ONLY JSON array:
[{"page": N, "topic": "Exact text unchanged", "level": 0}]`;

  try {
    const r = await ai.models.generateContent({
      model:    MODEL_NAME,
      contents: [{ text: prompt }],
      config:   { responseMimeType: 'application/json', responseSchema: INDEX_SCHEMA }
    });
    const selected = JSON.parse((r.text || '[]').replace(/```json|```/g, '').trim());
    if (Array.isArray(selected) && selected.length >= Math.min(5, max)) {
      return selected.sort((a, b) => a.page - b.page).slice(0, max);
    }
  } catch (err) {
    console.warn('Consolidation AI failed, using even distribution:', err.message);
  }

  // Fallback: even distribution
  return distributeEvenly(allEntries, max);
}

function distributeEvenly(entries, max) {
  if (entries.length <= max) return entries;
  const step = (entries.length - 1) / (max - 1);
  return Array.from({ length: max }, (_, i) => entries[Math.round(i * step)]);
}

// ─── Utilities ────────────────────────────────────────────────────────────────
const b64     = (dir, f) => fs.readFileSync(path.join(dir, f)).toString('base64');
const sleep   = ms       => new Promise(r => setTimeout(r, ms));
const elapsed = t0       => `${((Date.now() - t0) / 1000).toFixed(1)}s`;

// ─── Error handler ────────────────────────────────────────────────────────────
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE')
    return res.status(400).json({ error: `File must be under ${MAX_FILE_MB}MB.` });
  if (err) return res.status(400).json({ error: err.message });
  next();
});

// ─── Start ────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════════════╗
║   IndexGen Pro v5.0  ·  POLLING + QUALITY MODE      ║
║   Model  : ${MODEL_NAME.padEnd(42)}║
║   Chunks : ${String(CHUNK_SIZE + ' pages').padEnd(42)}║
║   Concurrent Gemini calls : ${String(CONCURRENT).padEnd(24)}║
║   1-page cap : ${String(ONE_PAGE_MAX + ' entries').padEnd(37)}║
║   STATUS: Zero timeout. Cold start safe.            ║
╚══════════════════════════════════════════════════════╝
`);
});
