""""Student Name : Eaint Taryar Linlat"""
###############################################################################
Task 10: Receipt Total Extraction with Gemini VLM
In this task, I used Google's Gemini 2.5 Flash vision-language model to extract the total price from a receipt image — a real-world document OCR problem.
What I did:

Loaded a receipt image using PIL, converted it to JPEG bytes, and sent it directly to Gemini via the google-genai client without any preprocessing or fine-tuning
Designed a structured extraction prompt that instructs Gemini to return only a plain float, with explicit examples ($19.74 → 19.74) and a NULL fallback if the total cannot be found — this removes ambiguity and reduces format errors
Parsed the response robustly — stripped currency symbols and non-numeric characters using regex, handled multiple decimal points by keeping only the last one, and returned None for failed extractions
Added retry logic with exponential back-off (waits 1s, then 2s) to handle temporary API errors gracefully
Built a small DataFrame (df_receipt) to support processing multiple receipt images in batch

Key lessons:

Prompt specificity directly affects output quality — without explicit format instructions and examples, the model may return "The total is $19.74" instead of "19.74", which breaks parsing
NULL handling is important — telling the model exactly what to return when it cannot find an answer prevents hallucinated totals that look plausible but are wrong
VLMs require no training for this task — Gemini reads and understands receipt layouts zero-shot, which would be impossible with traditional regex-based OCR on varied receipt formats
###############################################################################

import os
import io
import re
import time
from PIL import Image
from google import genai
from google.genai import types
from pathlib import Path
import pandas as pd

# ── 0. CONFIG — edit these two lines ─────────────────────────────────────────
GOOGLE_API_KEY = "AIzaSyDe0Hc-jPOS0LAAWdtYztLekVYlxKGRvy8"   # paste your Gemini API key here
RECEIPT_IMAGE  = "Task_10/receipt1.jpg"        # path to your receipt image

# ── 1. Initialise Gemini client ───────────────────────────────────────────────
client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL  = "gemini-2.5-flash"
print(f"Gemini client initialised — model: {MODEL}\n")

# ── 2. Load image ─────────────────────────────────────────────────────────────
if not os.path.exists(RECEIPT_IMAGE):
    raise FileNotFoundError(
        f"Could not find '{RECEIPT_IMAGE}'. "
        "Make sure it is in the same folder as Task_10.py."
    )

image = Image.open(RECEIPT_IMAGE).convert("RGB")
print(f"Loaded image: {RECEIPT_IMAGE}  ({image.size[0]}x{image.size[1]} px)")

# ── 3. Prompt ─────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = (
    "You are given a receipt image. "
    "Extract ONLY the final total amount (the amount the customer paid). "
    "Return ONLY the numerical value as a plain float — no currency symbols, "
    "no commas, no text, no explanation. "
    "For example, if the total is $19.74, respond with: 19.74\n"
    "For example, if the total is 38,000, respond with: 38000\n"
    "If you cannot determine the total, respond with: NULL"
)

# ── 4. Call Gemini ────────────────────────────────────────────────────────────
def predict_total(pil_image: Image.Image, max_retries: int = 2):
    """Send receipt image to Gemini; return (predicted_float, raw_response)."""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                    EXTRACTION_PROMPT,
                ],
            )
            raw = response.text.strip()

            if raw.upper() == "NULL":
                return None, raw

            # Keep only digits and dots, handle multiple dots
            cleaned = re.sub(r"[^\d.]", "", raw)
            if cleaned.count(".") > 1:
                parts   = cleaned.rsplit(".", 1)
                cleaned = parts[0].replace(".", "") + "." + parts[1]

            return (float(cleaned) if cleaned else None), raw

        except Exception as e:
            if attempt < max_retries:
                print(f"  [retry {attempt + 1}] {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"  [error] {e}")
                return None, str(e)

# ── 5. Run & display result ───────────────────────────────────────────────────
print("Sending receipt to Gemini …\n")
predicted_price, raw_response = predict_total(image)

print("=" * 45)
print("           RESULT")
print("=" * 45)
print(f"  Raw Gemini response : {raw_response}")
if predicted_price is not None:
    print(f"  Predicted total     : £/$ {predicted_price:.2f}")
else:
    print("  Predicted total     : Could not extract a price")
print("=" * 45)

# ── 6. Process multiple receipts ──────────────────────────────────────────────
local_dir = Path("Task_10")
paths = list(local_dir.glob("receipt1.*"))  # or receipt*.jpg
receipt_rows = []

for p in paths:
    img = Image.open(p).convert("RGB")
    # for demo, set known label manually or parse from filename
    total_price = None  # or a provided quantity
    receipt_rows.append({"image": img, "total_price": total_price})

df_receipt = pd.DataFrame(receipt_rows)
print(df_receipt)
print("\ndone")
