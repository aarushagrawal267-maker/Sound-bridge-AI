/**
 * SoundBridge AI — Google Sheet receiver.
 *
 * Paste this whole file into your Sheet: Extensions → Apps Script → replace everything → Save.
 * Change TOKEN below to any long random text, then Deploy → New deployment → Web app,
 * "Execute as: Me", "Who has access: Anyone" → Deploy → copy the Web app URL.
 *
 * Each SoundBridge store writes to its own tab (feedback, study_responses) with columns
 * timestamp | record_json. Requests without the right token are rejected.
 */
const TOKEN = "CHANGE-ME-to-a-long-random-text";

function tab_(name) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(name);
  if (!sh) {
    sh = ss.insertSheet(name);
    sh.appendRow(["timestamp", "record_json"]);
  }
  return sh;
}

function out_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function doPost(e) {
  const body = JSON.parse(e.postData.contents);
  if (body.token !== TOKEN) return out_({ ok: false, error: "bad token" });
  const name = String(body.table).replace(/[^a-z_]/g, "");
  const sh = tab_(name);
  if (body.action === "append") {
    sh.appendRow([new Date().toISOString(), JSON.stringify(body.record)]);
    return out_({ ok: true });
  }
  if (body.action === "load") {
    const values = sh.getDataRange().getValues().slice(1).map(r => r[1]).filter(String);
    return out_({ ok: true, rows: values });
  }
  if (body.action === "clear") {
    sh.clear();
    sh.appendRow(["timestamp", "record_json"]);
    return out_({ ok: true });
  }
  return out_({ ok: false, error: "unknown action" });
}
