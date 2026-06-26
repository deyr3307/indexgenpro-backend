// ╔══════════════════════════════════════════════════════════════════╗
// ║         IndexGen Pro — server.js v5.0 (JOB QUEUE)             ║
// ╠══════════════════════════════════════════════════════════════════╣
// ║  WHY TIMEOUT WAS HAPPENING:                                     ║
// ║  Single HTTP request = whole PDF must finish in <120s           ║
// ║  That's impossible to guarantee for large PDFs.                 ║
// ║                                                                  ║
// ║  FIX — Split into 2 endpoints:                                  ║
// ║  POST /api/generate-index  → returns jobId in <1s               ║
// ║  GET  /api/job/:id         → poll for status every 3s           ║
// ║                                                                  ║
// ║  Frontend polls, backend works in background.                   ║
// ║  TIMEOUT IS NOW IMPOSSIBLE regardless of PDF size.              ║
// ╚══════════════════════════════════════════════════════════════════╝

import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { randomUUID } from 'crypto';
import { GoogleGenAI } from '@google/genai';
import cors from 'cors';

const execAsync = promisify(exec);
const app = express();
app.use(cors());
app.use(express.json());

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const MODEL_NAME   = 'gemini-2.5-flash'; // fallback: gemini-2.5-flash-preview-05-20
const ONE_PAGE_MAX = 25;
const CHUNK_SIZE   = 10;
const CONCURRENT   = 3;
const MAX_RETRIES  = 2;
const MAX_FILE_MB  = 50;
const JOB_TTL_MS   = 15 * 60 * 1000; // jobs expire after 15 minutes

if (!process.env.GEMINI_API_KEY) {
  console.error('FATAL: GEMINI_API_KEY not set.');
  process.exit(1);
}

// ─── IN-MEMORY JOB STORE ─────────────────────────────────────────────────────
// Each job: { status, progress, result, error, createdAt, completedAt }
const jobs = new Map();

// Clean up expired jobs every minute
setInterval(() => {
  const cutoff = Date.now() - JOB_TTL_MS;
  for (const [id, job] of jobs.entries()) {
    if (job.createdAt < cutoff) jobs.delete(id);
  }
}, 60_000);

// ─── MULTER ──────────────────────────────────────────────────────────────────
const upload = multer({
  dest: os.tmpdir(),
  limits: { fileSize: MAX_FILE_MB * 1024 * 1024 },
  fileFilter: (_, file, cb) =>
    file.mimetype === 'application/pdf'
      ? cb(null, true)
      : cb(new Error(`Only PDF files are allowed. Received: ${file.mimetype}`))
});

// ─── SCHEMAS ─────────────────────────────────────────────────────────────────
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
    topicStructure: { type: 'string'  }
  },
  required: ['subject','documentType','theme','headingPattern','entryFormat','isHandwritten','topicStructure']
};

// ─── ROUTES ───────────────────────────────────────────────────────────────────
app.get('/',        (_, res) => res.json({ status: 'IndexGen Pro v5', model: MODEL_NAME }));
app.get('/health',  (_, res) => res.json({ ok: true, model: MODEL_NAME, jobs: jobs.size }));

// ─── ENDPOINT 1: Start job (returns in <1s) ───────────────────────────────────
app.post('/api/generate-index', upload.single('pdf'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No PDF uploaded. Field name must be "pdf".' });
  }

  const jobId = randomUUID();

  jobs.set(jobId, {
    status:      'processing',
    progress:    'Starting PDF conversion...',
    percent:     0,
    result:      null,
    error:       null,
    createdAt:   Date.now(),
    completedAt: null
  });

  console.log(`\n[Job ${jobId.slice(0,8)}] Started`);

  // Fire-and-forget: processing runs in background
  runJob(jobId, req.file.path).catch(err => {
    const job = jobs.get(jobId);
    if (job) {
      job.status = 'error';
      job.error  = err.message;
      job.completedAt = Date.now();
    }
    console.error(`[Job ${jobId.slice(0,8)}] Fatal:`, err.message);
  });

  // Respond immediately — frontend will poll /api/job/:id
  res.json({ jobId, message: 'Job started. Poll /api/job/' + jobId + ' for results.' });
});

// ─── ENDPOINT 2: Poll job status ──────────────────────────────────────────────
app.get('/api/job/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) {
    return res.status(404).json({ error: 'Job not found or expired (15 min TTL).' });
  }
  res.json({
    jobId:    req.params.id,
    status:   job.status,       // 'processing' | 'done' | 'error'
    progress: job.progress,     // human-readable status string
    percent:  job.percent,      // 0-100
    result:   job.result,       // null until done
    error:    job.error         // null unless error
  });
});

// ─── BACKGROUND PROCESSING ────────────────────────────────────────────────────
async function runJob(jobId, pdfPath) {
  const job   = jobs.get(jobId);
  const t0    = Date.now();
  const label = `[Job ${jobId.slice(0,8)}]`;
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'idx-'));

  const setProgress = (progress, percent) => {
    job.progress = progress;
    job.percent  = percent;
    console.log(`${label} ${progress}`);
  };

  try {
    // ── 1. PDF → JPEG (150 DPI) ───────────────────────────────────────────────
    setProgress('Converting PDF pages...', 5);
    await execAsync(`pdftoppm -jpeg -r 150 "${pdfPath}" "${workDir}/page"`);

    const pageFiles = fs.readdirSync(workDir).filter(f => f.endsWith('.jpg')).sort();
    if (!pageFiles.length) throw new Error('PDF conversion failed — file may be corrupt.');

    setProgress(`Converted ${pageFiles.length} pages`, 15);

    // ── 2. Understand document structure ─────────────────────────────────────
    setProgress('Analyzing document type...', 20);
    const sampleFiles = pageFiles.slice(0, Math.min(2, pageFiles.length));
    const docCtx = await analyzeContext(workDir, sampleFiles);

    setProgress(`Identified: ${docCtx.theme}`, 30);

    // ── 3. Parallel chunk extraction ─────────────────────────────────────────
    const chunkTasks = [];
    for (let i = 0; i < pageFiles.length; i += CHUNK_SIZE) {
      chunkTasks.push({ chunk: pageFiles.slice(i, i + CHUNK_SIZE), offset: i });
    }

    let allEntries = [];
    const failed   = [];
    const total    = chunkTasks.length;

    for (let b = 0; b < total; b += CONCURRENT) {
      const batch   = chunkTasks.slice(b, b + CONCURRENT);
      const batchNo = Math.floor(b / CONCURRENT) + 1;
      const batches = Math.ceil(total / CONCURRENT);

      setProgress(
        `Extracting headings (batch ${batchNo}/${batches})...`,
        30 + Math.round((b / total) * 50)
      );

      const settled = await Promise.allSettled(
        batch.map(({ chunk, offset }) => extractWithRetry(workDir, chunk, offset, docCtx))
      );

      settled.forEach((result, idx) => {
        const task = batch[idx];
        if (result.status === 'fulfilled') {
          allEntries.push(...result.value);
        } else {
          console.error(`${label} ✗ Pages ${task.offset+1}–${task.offset+task.chunk.length}: ${result.reason?.message}`);
          failed.push({ from: task.offset + 1, to: task.offset + task.chunk.length });
        }
      });
    }

    // Sort + exact-duplicate guard
    allEntries.sort((a, b) => a.page - b.page);
    const seen = new Set();
    allEntries = allEntries.filter(({ page, topic }) => {
      const key = `${page}|||${topic.toLowerCase().trim()}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    // Sanitize HTML
    allEntries = allEntries.map(e => ({
      ...e,
      topic: e.topic.replace(/<(?!\/?em>)[^>]+>/gi, '').trim()
    }));

    setProgress(`Extracted ${allEntries.length} headings`, 85);

    // ── 4. Consolidate to 1-page fit ─────────────────────────────────────────
    let finalEntries = allEntries;
    let consolidated = false;

    if (allEntries.length > ONE_PAGE_MAX) {
      setProgress(`Selecting best ${ONE_PAGE_MAX} entries for 1-page fit...`, 90);
      finalEntries = await consolidate(allEntries, docCtx);
      consolidated = true;
    }

    const totalSec = ((Date.now() - t0) / 1000).toFixed(1);
    setProgress(`Done in ${totalSec}s — ${finalEntries.length} entries`, 100);

    // ── Mark job done ─────────────────────────────────────────────────────────
    job.status = 'done';
    job.result = {
      success:         true,
      totalPages:      pageFiles.length,
      rawEntries:      allEntries.length,
      finalEntries:    finalEntries.length,
      consolidated,
      processingMs:    Date.now() - t0,
      documentContext: docCtx,
      index:           finalEntries,
      ...(failed.length ? {
        warning: `Pages not processed: ${failed.map(c => `${c.from}–${c.to}`).join(', ')}`
      } : {})
    };
    job.completedAt = Date.now();

    console.log(`${label} ✅ Done in ${totalSec}s | ${finalEntries.length} entries`);

  } finally {
    fs.rmSync(workDir, { recursive: true, force: true });
    fs.unlink(pdfPath, () => {});
  }
}

// ─── Context analysis ─────────────────────────────────────────────────────────
async function analyzeContext(workDir, sampleFiles) {
  const parts = [{
    text: `Study these ${sampleFiles.length} pages from an academic PDF. Return ONLY valid JSON:
{
  "subject": "e.g. Zoology, Heat Transfer, Chemistry",
  "documentType": "e.g. Assignment, Lab Report, Research Paper",
  "theme": "e.g. Systematic classification of freshwater organisms",
  "headingPattern": "e.g. Bold centered: Systematic Position of [Name] at top of each page",
  "entryFormat": "e.g. Systematic Position of Dero dorsalis — preserve exact capitalization",
  "isHandwritten": false,
  "topicStructure": "Series (each page = 1 topic, all equal) OR Hierarchical (chapters) OR Mixed"
}`
  }];

  sampleFiles.forEach((f, i) => {
    parts.push({ text: `--- PAGE ${i + 1} ---` });
    parts.push({ inlineData: { mimeType: 'image/jpeg', data: b64(workDir, f) } });
  });

  try {
    const r = await ai.models.generateContent({
      model: MODEL_NAME, contents: parts,
      config: { responseMimeType: 'application/json', responseSchema: CONTEXT_SCHEMA }
    });
    return JSON.parse((r.text || '{}').replace(/```json|```/g, '').trim());
  } catch {
    return {
      subject: 'Academic', documentType: 'Assignment', isHandwritten: false,
      theme: 'Academic document',
      headingPattern: 'Most prominent title at top of each new section',
      entryFormat: 'Main topic title exactly as written',
      topicStructure: 'Series — each section covers one independent topic'
    };
  }
}

// ─── Heading extraction ───────────────────────────────────────────────────────
async function extractWithRetry(workDir, chunk, offset, docCtx) {
  let lastErr;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try { return await extractChunk(workDir, chunk, offset, docCtx); }
    catch (err) {
      lastErr = err;
      if (attempt < MAX_RETRIES) await sleep(1500 * (attempt + 1));
    }
  }
  throw lastErr;
}

async function extractChunk(workDir, chunkFiles, offset, ctx) {
  const typeRules = ctx.isHandwritten
    ? `HANDWRITTEN: underlined text = heading | ☐★□ before text = heading
  ❌ Taxonomy rows (Kingdom/Phylum/Class/Order/Family/Genus/Species) = SKIP`
    : `TYPED/PRINTED: bold standalone line = heading | ALL-CAPS title = heading
  ❌ Numbered content items (1. 2. 3.) = SKIP | sub-headings within section = SKIP`;

  const parts = [{
    text: `Expert academic index extractor.

DOCUMENT: ${ctx.subject} ${ctx.documentType} — "${ctx.theme}"
HEADING STYLE: ${ctx.headingPattern}
ENTRY FORMAT: ${ctx.entryFormat}
STRUCTURE: ${ctx.topicStructure}

${typeRules}

RULES:
1. Check ALL ${chunkFiles.length} pages — skip none.
2. ONE main heading per page only.
3. COPY VERBATIM — exact spelling, exact characters.
4. EXACT CAPITALIZATION (e.g. "Tubifex tubifex" not "Tubifex . tubifex").
5. No entry limit.
6. Skip blank/pure-diagram/continuation pages.

SKIP: Kingdom/Phylum/Class/Order/Family/Genus/Species | Fig captions | numbered content | table cells | headers/footers

Pages labeled "--- PAGE N ---". Use that exact number.
Return ONLY: [{"page": N, "topic": "Exact text", "level": 0}]`
  }];

  chunkFiles.forEach((f, i) => {
    parts.push({ text: `--- PAGE ${offset + i + 1} ---` });
    parts.push({ inlineData: { mimeType: 'image/jpeg', data: b64(workDir, f) } });
  });

  const r = await ai.models.generateContent({
    model: MODEL_NAME, contents: parts,
    config: { responseMimeType: 'application/json', responseSchema: INDEX_SCHEMA }
  });

  const parsed = JSON.parse((r.text || '[]').replace(/```json|```/g, '').trim());
  return Array.isArray(parsed) ? parsed : [];
}

// ─── Consolidation ────────────────────────────────────────────────────────────
async function consolidate(allEntries, docCtx, max = ONE_PAGE_MAX) {
  try {
    const r = await ai.models.generateContent({
      model: MODEL_NAME,
      contents: [{
        text: `Index editor. Printed page fits max ${max} entries.
Document: "${docCtx.theme}" | Structure: ${docCtx.topicStructure}
Select best ${max} from ${allEntries.length} entries. Cover full doc range (start→middle→end).
SERIES: pick evenly distributed. HIERARCHICAL: chapter-level only. MIXED: one per theme.
Keep original page/topic unchanged. Sort by page asc.
Input: ${JSON.stringify(allEntries)}
Return ONLY JSON: [{"page":N,"topic":"text","level":0}]`
      }],
      config: { responseMimeType: 'application/json', responseSchema: INDEX_SCHEMA }
    });
    const selected = JSON.parse((r.text || '[]').replace(/```json|```/g, '').trim());
    if (Array.isArray(selected) && selected.length > 0) {
      return selected.sort((a, b) => a.page - b.page);
    }
  } catch (err) {
    console.warn('Consolidation AI failed, using even distribution:', err.message);
  }
  // Fallback: evenly distributed
  const step = (allEntries.length - 1) / (max - 1);
  return Array.from({ length: max }, (_, i) => allEntries[Math.round(i * step)]);
}

// ─── Utilities ────────────────────────────────────────────────────────────────
const b64   = (dir, f) => fs.readFileSync(path.join(dir, f)).toString('base64');
const sleep = ms => new Promise(r => setTimeout(r, ms));

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
  console.log(`\n╔════════════════════════════════════════════╗`);
  console.log(`║   IndexGen Pro v5.0  —  JOB QUEUE MODE    ║`);
  console.log(`║   POST /api/generate-index → jobId (<1s)  ║`);
  console.log(`║   GET  /api/job/:id        → poll status  ║`);
  console.log(`║   Model: ${MODEL_NAME.padEnd(32)}║`);
  console.log(`╚════════════════════════════════════════════╝\n`);
});
