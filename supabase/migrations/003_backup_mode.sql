CREATE TABLE IF NOT EXISTS public.backup_settings (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    enabled BOOLEAN NOT NULL DEFAULT false,
    reference_label TEXT CHECK (reference_label IN ('authentic', 'ai_generated')),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (enabled = true OR reference_label IS NULL)
);

CREATE TABLE IF NOT EXISTS public.backup_evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id UUID,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    reference_label TEXT NOT NULL CHECK (reference_label IN ('authentic', 'ai_generated')),
    model_prediction TEXT NOT NULL CHECK (model_prediction IN ('real', 'ai_generated', 'needs_review')),
    model_probability NUMERIC(5, 2) NOT NULL,
    is_match BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.backup_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.backup_evaluations ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users manage own backup settings" ON public.backup_settings;
CREATE POLICY "Users manage own backup settings" ON public.backup_settings
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users view own backup evaluations" ON public.backup_evaluations;
CREATE POLICY "Users view own backup evaluations" ON public.backup_evaluations
    FOR SELECT USING (auth.uid() = user_id);
