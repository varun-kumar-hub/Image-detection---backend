CREATE TABLE IF NOT EXISTS user_ai_settings (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    gemini_api_key_encrypted TEXT,
    gemini_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    gemini_model TEXT NOT NULL DEFAULT 'gemini-2.5-flash',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE user_ai_settings ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own AI settings" ON user_ai_settings;
CREATE POLICY "Users manage own AI settings" ON user_ai_settings FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
