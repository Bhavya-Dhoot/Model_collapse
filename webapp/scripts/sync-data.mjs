/* Copy the single source of truth (demo/data.json, produced by
   code/export_dashboard_data.py from results/main.csv) into src/ so Vite can
   import it. Runs automatically before dev and build, so the webapp can never
   drift from the numbers the paper and the offline dashboard use. */
import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const src = resolve(here, '../../demo/data.json')
const dest = resolve(here, '../src/data.json')

if (!existsSync(src)) {
  console.error(`[sync-data] missing ${src}\n  run: python code/export_dashboard_data.py`)
  process.exit(1)
}
mkdirSync(dirname(dest), { recursive: true })
copyFileSync(src, dest)
console.log(`[sync-data] ${(statSync(dest).size / 1024).toFixed(0)} KB -> src/data.json`)
