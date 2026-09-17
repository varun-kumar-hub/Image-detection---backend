ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS analysis_key TEXT UNIQUE;
ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS storage_path TEXT;
ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS filename TEXT;
ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS confidence_explanation TEXT;
ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS model_name TEXT;
ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS image_info JSONB DEFAULT '{}'::jsonb;
ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS manipulation JSONB DEFAULT '{}'::jsonb;
ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS explanation JSONB DEFAULT '{}'::jsonb;
ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS gradcam JSONB DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_analysis_results_analysis_key ON analysis_results(analysis_key);
