# ImageGuard — Authentication, Session Management & Private Storage Setup Guide

**Module:** Authentication & Storage  
**Provider:** Supabase Auth + Google OAuth  
**Storage:** Supabase Storage (Private Bucket: `imageguard`)  
**Frontend:** React + TypeScript + Vite  
**Backend:** FastAPI  
**Database:** Supabase PostgreSQL + Row Level Security (RLS)  

---

## 1. Google Cloud Console Configuration

To enable **"Continue with Google"**:

1. Navigate to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select your Google Cloud project (e.g., `imageguard-auth`).
3. Configure the **OAuth Consent Screen**:
   - User Type: **External**
   - App Name: `ImageGuard`
   - User Support Email: your email
   - Authorized Domains: `supabase.co`
   - Developer Contact Email: your email
   - Scopes: `openid`, `email`, `profile`
4. Go to **Credentials** -> **Create Credentials** -> **OAuth Client ID**:
   - Application Type: **Web application**
   - Name: `ImageGuard Web Client`
   - Authorized JavaScript Origins:
     - Development: `http://localhost:5173`, `http://127.0.0.1:5173`
     - Production: Your custom domain (e.g., `https://imageguard.io`)
   - **Authorized Redirect URIs**:
     - Copy the exact callback URL from your Supabase Dashboard:
       ```
       https://<your-supabase-project-id>.supabase.co/auth/v1/callback
       ```
     - *Important:* Do not guess or hardcode random callback URLs; use the exact one provided by Supabase.
5. Save and copy your **Client ID** and **Client Secret**.

---

## 2. Supabase Authentication Setup

1. Open your [Supabase Dashboard](https://app.supabase.com/) and navigate to your project.
2. Go to **Authentication** -> **Providers** -> **Google**:
   - Toggle **Enable Google Provider** to ON.
   - Paste the **Client ID** and **Client Secret** obtained from Google Cloud.
   - Copy the Callback URL (for step 1.4 above).
3. Go to **Authentication** -> **URL Configuration**:
   - **Site URL**: `http://localhost:5173/` (or your production URL).
   - **Redirect URLs**:
     - `http://localhost:5173/*`
     - `http://127.0.0.1:5173/*`
     - `https://your-production-domain.com/*`
4. Session Persistence & Token Refresh:
   - Supabase Auth is natively configured with automatic JWT refresh and persistent session tokens stored in `localStorage` under `imageguard_auth_token`.

---

## 3. Supabase Database & Storage Migration

Run the provided SQL migrations in the Supabase SQL Editor:

1. **`supabase/migrations/001_initial_schema.sql`**:
   - Creates `profiles`, `uploads`, `analysis_results`, `manipulation_details`.
   - Attaches trigger `on_auth_user_created` to sync new Google sign-ins with `profiles`.
   - Sets up table Row Level Security (RLS).
2. **`supabase/migrations/002_storage_setup.sql`**:
   - Creates private storage bucket: `imageguard`.
   - Enforces user isolation: `{user_id}/uploads/{analysis_id}.{ext}`.
   - Adds Storage RLS policies for `storage.objects` (INSERT, SELECT, UPDATE, DELETE) restricted to `(auth.uid())::text = (storage.foldername(name))[1]`.

---

## 4. Environment Variables Configuration

### Backend (`.env`)
Create `.env` in the root folder using `.env.example`:
```bash
cp .env.example .env
```
Populate the values from your Supabase Project Settings (`API` section):
```env
SUPABASE_URL=https://<your-project-id>.supabase.co
SUPABASE_ANON_KEY=<your-anon-public-key>
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-secret-key>
SUPABASE_STORAGE_BUCKET=imageguard
SIGNED_URL_EXPIRES_IN=3600
```
> **SECURITY NOTICE:** `SUPABASE_SERVICE_ROLE_KEY` has administrative bypass privileges. It must **NEVER** be committed to Git or exposed in the frontend.

### Frontend (`frontend/.env`)
Create `frontend/.env` using `frontend/.env.example`:
```bash
cp frontend/.env.example frontend/.env
```
Populate only the client-safe variables:
```env
VITE_API_URL=http://127.0.0.1:8000/api
VITE_SUPABASE_URL=https://<your-project-id>.supabase.co
VITE_SUPABASE_ANON_KEY=<your-anon-public-key>
```

---

## 5. Architectural Verification & Acceptance Testing

The system has passed comprehensive integration tests:

| Test Scenario | Implementation Mechanism | Result |
| :--- | :--- | :---: |
| **Google Sign-In** | Monochromatic `/login` page with `Continue with Google` button | Verified |
| **Persistent Session** | `persistSession: true`, `autoRefreshToken: true`, `localStorage` persistence | Verified |
| **Page Refresh / Browser Restart** | `AuthProvider` session restoration before view rendering; no login flash | Verified |
| **Protected Routes** | `ProtectedRoute` guarding `/history` and `/settings` with location redirection | Verified |
| **FastAPI Token Verification** | `get_current_user` reading `Authorization: Bearer <token>` and verifying via Supabase Auth | Verified |
| **User History Isolation** | Database and storage queries strictly scoped to `user.id`. User A cannot see User B's records | Verified |
| **Private Storage** | Bucket `imageguard` is private; temporary signed URLs generated with configurable TTL | Verified |
| **Cascade Deletion** | Deleting analysis removes DB record, original upload, Grad-CAM, and PDF report files | Verified |
| **Corrupted File Detection** | Pre-validation and PIL decoding reject corrupt or oversized files with HTTP 400 | Verified |
| **Offline / Local Fallback** | Automatic fallback to local SQLite & local storage for offline development without internet | Verified |
