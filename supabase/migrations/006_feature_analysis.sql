ALTER TABLE analysis_results
    ADD COLUMN IF NOT EXISTS feature_analysis JSONB DEFAULT '{}'::jsonb;
