from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import os
from email_alert import send_email_alert

def job_scan_and_alert():
    # 1) เรียกฟังก์ชัน run_scan ของเรา เพื่อสแกน universe (ตัวอย่าง: SAMPLE_MICRO / uploaded CSV)
    try:
        tickers = SAMPLE_MICRO  # หรือ load from CSV path / repo
        out, details = run_scan(tickers)  # ใช้ฟังก์ชันที่เขียนไว้ใน app
        # 2) filter top results (เกณฑ์ค่า score)
        threshold = float(os.environ.get("ALERT_SCORE_THRESHOLD", 80))
        top_hits = out[out['score']>=threshold]
        if not top_hits.empty:
            lines = []
            for i, r in top_hits.iterrows():
                lines.append(f"{r['ticker']} — score {r['score']}")
            body = "Daily MCRF Alert\n\n" + "\n".join(lines)
            send_email_alert("MCRF Daily Alert", body)
            print("Alert sent")
    except Exception as e:
        print("Scheduled job error", e)

# Start scheduler only once
if os.environ.get("ENABLE_SCHEDULER","true").lower() in ["true","1","yes"]:
    scheduler = BackgroundScheduler()
    # run at 9:00 UTC daily (adjust timezone as needed) — CRON or interval can be used
    scheduler.add_job(job_scan_and_alert, 'interval', hours=24, next_run_time=None)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown(wait=False))
