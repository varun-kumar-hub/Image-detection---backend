-- Supabase Storage & Schema Setup for ImageGuard (v4.0)
-- Private Storage Bucket 'imageguard' and Row Level Security (RLS) Policies

-- 1. Create the private 'imageguard' bucket if it does not already exist
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'imageguard',
    'imageguard',
    false,
    10485760, -- 10 MB limit
    ARRAY['image/jpeg', 'image/png', 'image/webp', 'application/pdf']
)
ON CONFLICT (id) DO UPDATE SET
    public = false,
    file_size_limit = 10485760,
    allowed_mime_types = ARRAY['image/jpeg', 'image/png', 'image/webp', 'application/pdf'];

-- 2. Extend uploads table with additional metadata fields if missing
ALTER TABLE IF EXISTS uploads
    ADD COLUMN IF NOT EXISTS width INTEGER,
    ADD COLUMN IF NOT EXISTS height INTEGER,
    ADD COLUMN IF NOT EXISTS file_extension TEXT;

-- 3. Extend analysis_results table with processed artifact paths
ALTER TABLE IF EXISTS analysis_results
    ADD COLUMN IF NOT EXISTS gradcam_path TEXT,
    ADD COLUMN IF NOT EXISTS ela_path TEXT,
    ADD COLUMN IF NOT EXISTS report_path TEXT;

-- 4. Enable Row Level Security on storage.objects if not already enabled
ALTER TABLE storage.objects ENABLE ROW LEVEL SECURITY;

-- 5. Storage RLS Policies: Authenticated Users only access their own user-scoped folder:
-- Pattern: imageguard/{user_id}/...

-- Allow users to upload to their own folder: imageguard/{user_id}/*
DROP POLICY IF EXISTS "Users can upload their own images" ON storage.objects;
CREATE POLICY "Users can upload their own images"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (
    bucket_id = 'imageguard'
    AND (auth.uid())::text = (storage.foldername(name))[1]
);

-- Allow users to select/view their own images: imageguard/{user_id}/*
DROP POLICY IF EXISTS "Users can read their own images" ON storage.objects;
CREATE POLICY "Users can read their own images"
ON storage.objects FOR SELECT
TO authenticated
USING (
    bucket_id = 'imageguard'
    AND (auth.uid())::text = (storage.foldername(name))[1]
);

-- Allow users to update their own images: imageguard/{user_id}/*
DROP POLICY IF EXISTS "Users can update their own images" ON storage.objects;
CREATE POLICY "Users can update their own images"
ON storage.objects FOR UPDATE
TO authenticated
USING (
    bucket_id = 'imageguard'
    AND (auth.uid())::text = (storage.foldername(name))[1]
);

-- Allow users to delete their own images: imageguard/{user_id}/*
DROP POLICY IF EXISTS "Users can delete their own images" ON storage.objects;
CREATE POLICY "Users can delete their own images"
ON storage.objects FOR DELETE
TO authenticated
USING (
    bucket_id = 'imageguard'
    AND (auth.uid())::text = (storage.foldername(name))[1]
);
