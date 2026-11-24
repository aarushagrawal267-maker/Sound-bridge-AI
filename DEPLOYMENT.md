# Deploying SoundBridge AI for the pilot study

This guide puts SoundBridge AI online on **Streamlit Community Cloud** (free). Participant responses go to a
**Google Sheet you own**, so they survive app restarts. It takes about 20 minutes. You need a GitHub account (the
repo is already there) and a Google account.

---

## Easiest option: Apps Script receiver (no Google Cloud, about 5 min)
1. Create a blank Google Sheet.
2. Open **Extensions → Apps Script**, delete everything there, and paste the contents of
   [`deploy/google_apps_script.gs`](deploy/google_apps_script.gs). Set `TOKEN` to a long random text, then click
   **Save**.
3. Click **Deploy → New deployment → ⚙ → Web app**. Set *Execute as*: **Me** and *Who has access*: **Anyone**, then
   click **Deploy**. Allow the permission prompt (Advanced → Go to project → Allow).
4. Copy the **Web app URL** (it ends in `/exec`).
5. In Streamlit secrets put:
   ```toml
   admin_password = "your-researcher-password"
   [storage]
   webhook_url = "https://script.google.com/macros/s/.../exec"
   token = "the same TOKEN as in the script"
   ```
You can then skip Steps 1–2 below and go straight to Step 3 (deploy).

---

## Alternative: service account

## Step 1: Create the Google Sheet (2 min)
1. Go to <https://sheets.google.com> and create a blank sheet named, for example, `SoundBridge pilot responses`.
2. Copy its URL from the address bar. You will need it in Step 3.

SoundBridge creates two tabs in it automatically: `feedback` and `study_responses`.

## Step 2: Create a Google service account (8 min)
This is the "robot" account the app uses to write into your sheet.
1. Open <https://console.cloud.google.com/>, create a project (for example, `soundbridge`) and select it.
2. **APIs & Services → Library:** enable **Google Sheets API** and **Google Drive API**.
3. **IAM & Admin → Service Accounts → Create service account.** Name it `soundbridge` and skip the optional steps.
4. Open the new account → **Keys → Add key → Create new key → JSON**. A `.json` file downloads. Keep it private.
5. Copy the account's e-mail (it looks like `soundbridge@your-project.iam.gserviceaccount.com`).
6. Back in your Google Sheet, click **Share**, paste that e-mail and give it **Editor** access.

## Step 3: Deploy on Streamlit Community Cloud (5 min)
1. Go to <https://share.streamlit.io>, sign in with GitHub and click **Create app → Deploy a public app from GitHub**.
2. Fill in:
   * **Repository:** `aarushagrawal267-maker/sound-bridge-ai`
   * **Branch:** `main`
   * **Main file path:** `app.py`
   * **App URL:** choose one, for example `soundbridge-ai`
3. Open **Advanced settings**:
   * **Python version:** 3.12
   * **Secrets:** paste the contents of [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) and
     fill in:
     * `admin_password`: a long password only you know
     * `sheet_url`: the URL from Step 1
     * every field under `[gcp_service_account]`, copied from the JSON key file. Keep the `\n` sequences inside
       `private_key` exactly as they are in the file.
4. Click **Deploy**.

The first start installs packages, downloads the dataset and trains the model. This takes about **5–10 minutes**,
and the app shows its progress. Later starts reuse the build while the app stays awake.

## Step 4: Check that it works (3 min)
1. Open the app, load an example taste on Home, go to **Music Discovery**, switch on *Store my feedback* and like
   one track.
2. Check your Google Sheet. A `feedback` tab should now contain one row.
3. In the app sidebar, open **Researcher login** and enter your admin password. The Feedback Dashboard should show
   the event, and **Delete all stored feedback** removes it again (this also clears the sheet tab).

If the sheet stays empty, the dashboard header says `stored in local file`. In that case the secrets are
incomplete or the sheet is not shared with the service-account e-mail.

## Step 5: Run the pilot
* **Before each session**, open the app yourself so it is awake. Community Cloud apps sleep after a period without
  visitors, and waking one can trigger the 5–10 minute first-start build again.
* **Give participants this link:** `https://<your-app-url>.streamlit.app/Pilot_Study`
* **Each participant:**
  1. agrees to the consent statement
  2. adds 2–5 songs they genuinely like on **Music Discovery**
  3. returns to **Pilot Study**, rates List A and List B, and submits

  List order is randomised, and they see which list was which only after submitting.
* **Target:** 10–20 participants. Answers are anonymous: a random ID, no names or e-mails.
* **Results:** the **Pilot Study → Results report** tab updates live and offers a Markdown report download. Raw
  answers are in the `study_responses` tab of your sheet.

## Good to know
* **Participants cannot** delete data, export raw events or run experiments. Those need the researcher login.
* **Privacy:** tell participants their anonymous answers are stored in a Google Sheet you control.
* **Memory:** if Community Cloud reports the app ran out of resources during the first build, reboot it from the
  app menu. The build resumes from the downloaded dataset.
* **Install fallback:** if the package installer struggles with the CPU-only PyTorch line, Community Cloud retries
  with `pip` automatically.
* **Running locally** with the sheet: save the same secrets as `.streamlit/secrets.toml` (never commit it; it is in
  `.gitignore`).
