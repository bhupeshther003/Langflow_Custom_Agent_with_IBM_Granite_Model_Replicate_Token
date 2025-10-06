# replicate_granite_component.py

import os
import time
import requests
from typing import Any, Optional

from langflow.custom.custom_component.component import Component
from langflow.io import MessageTextInput, Output
from langflow.schema.data import Data


class ReplicateGranite(Component):
    display_name = "Replicate Granite LLM"
    description = "Call IBM Granite (ibm-granite/granite-3.2-8b-instruct) on Replicate using version UUID."
    documentation = "https://replicate.com/ibm-granite/granite-3.2-8b-instruct"
    icon = "bot"
    name = "ReplicateGranite"

    inputs = [
        MessageTextInput(
            name="prompt",
            display_name="Prompt",
            info="Input prompt for the model",
            value="Hello Granite!",
            tool_mode=True,
        ),
        MessageTextInput(                                  
            name="replicate_api_token",
            display_name="Replicate API Token",
            info="If left blank, will use REPLICATE_API_TOKEN from env",
            value="YourRelicateAPITokenHere",
            tool_mode=True,
        ),
        MessageTextInput(
            name="model_version",
            display_name="Model Version UUID",
            info="Replicate model version id (e.g. 7kqz6aczz5rme0cns1rrsj94yr)",
            value="a325a0cacfb0aa9226e6bad1abe5385f1073f4c7f8c36e52ed040e5409e6c034",
            tool_mode=True,
        ),
        MessageTextInput(
            name="timeout_seconds",
            display_name="Timeout (s)",
            info="Max seconds to wait for prediction",
            value="90",
            tool_mode=True,
        ),
        MessageTextInput(
            name="poll_interval",
            display_name="Poll interval (s)",
            info="Seconds between polling requests",
            value="2",
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(display_name="Generated Text", name="text", method="build_output"),
    ]

    def _extract_text(self, output: Any) -> Optional[str]:
        """Extract text from various output formats returned by Replicate."""
        if output is None:
            return None
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, (int, float, bool)):
            return str(output)
        if isinstance(output, list):
            for item in reversed(output):
                t = self._extract_text(item)
                if t:
                    return t
            return None
        if isinstance(output, dict):
            for key in ("generated_text", "text", "content", "output", "caption"):
                if key in output:
                    t = self._extract_text(output[key])
                    if t:
                        return t
            for v in output.values():
                t = self._extract_text(v)
                if t:
                    return t
        return str(output)

    def build_output(self) -> Data:
        token = (self.replicate_api_token or "").strip() or os.getenv("REPLICATE_API_TOKEN")
        if not token:
            return Data(value="❌ Error: No Replicate API token provided.")

        version = (self.model_version or "").strip()
        if not version:
            return Data(value="❌ Error: No model version UUID provided.")

        try:
            timeout = float(self.timeout_seconds)
        except Exception:
            timeout = 90.0
        try:
            poll_interval = float(self.poll_interval)
        except Exception:
            poll_interval = 2.0

        create_url = "https://api.replicate.com/v1/predictions"
        headers = {
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "version": version,   # ✅ Use only version UUID
            "input": {"prompt": self.prompt},
        }

        # 1. Create prediction
        try:
            resp = requests.post(create_url, headers=headers, json=payload, timeout=15)
        except Exception as e:
            return Data(value=f"❌ Exception while creating prediction: {e}")

        if resp.status_code not in (200, 201):
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            return Data(value=f"❌ Error creating prediction: {resp.status_code} - {err}")

        try:
            pred = resp.json()
        except Exception as e:
            return Data(value=f"❌ Prediction created but JSON parse failed: {e} - raw: {resp.text}")

        pred_id = pred.get("id")
        if not pred_id:
            return Data(value=f"❌ Prediction response missing id: {pred}")

        # 2. Poll until finished
        status = pred.get("status")
        poll_url = f"{create_url}/{pred_id}"
        start = time.time()
        detail = None

        while status not in ("succeeded", "failed") and (time.time() - start) < timeout:
            time.sleep(poll_interval)
            try:
                r = requests.get(poll_url, headers=headers, timeout=15)
            except Exception as e:
                detail = f"Exception while polling: {e}"
                break

            if r.status_code != 200:
                detail = f"Polling error {r.status_code}: {r.text}"
                break

            try:
                pred = r.json()
            except Exception as e:
                detail = f"Polling JSON error: {e} - raw: {r.text}"
                break

            status = pred.get("status")

        # 3. Final status
        if status == "succeeded":
            output = pred.get("output")
            extracted = self._extract_text(output)
            if extracted:
                data_obj = Data(value=extracted)
                self.status = data_obj
                return data_obj
            return Data(value=f"⚠️ Prediction succeeded but no text extracted. Raw: {output}")

        else:
            err = pred.get("error") if isinstance(pred, dict) else None
            logs = pred.get("logs") if isinstance(pred, dict) else None
            reason = detail or err or logs or f"status={status}"
            return Data(value=f"❌ Prediction failed: {reason}")

