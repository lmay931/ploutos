import json
import time
from google.genai import types

from core.api_client import gemini_client
from core.config import (
    GEMINI_MODEL_NAME, 
    MAX_RETRIES, 
    RETRY_DELAY_SECONDS, 
    EXPERT_RESPONSE_SCHEMA
)

class PortfolioModel:
    def __init__(self):
        self.system_instruction = """
        You are an automated portfolio allocation engine.
        You receive directional signals from independent sub-models analyzing specific asset data.
        Your function is to evaluate these sub-model outputs, synthesize a consensus, and output a final directional execution signal.
        
        PARAMETERS:
        - CONSENSUS: If models align on a directional bias, return that action.
        - DIVERGENCE: If models diverge, weigh fundamental structural data against near-term sentiment catalysts.
        - PRESERVATION: If signals lack conviction, or if market conditions are highly ambiguous, default to capital preservation. Return HOLD.
        """

    def evaluate(self, ticker, date, model_outputs):
        outputs_text = ""
        for model_name, output in model_outputs.items():
            outputs_text += f"\n[{model_name}]\n- Action: {output['action']}\n- Reasoning: {output['reasoning']}\n"

        prompt = f"""
        ASSET: {ticker}
        DATE: {date}
        
        SUB-MODEL SIGNALS:{outputs_text}
        
        Evaluate the provided signals and output the final execution decision.
        """

        for attempt in range(MAX_RETRIES):
            try:
                response = gemini_client.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction, 
                        response_mime_type="application/json",
                        response_schema=EXPERT_RESPONSE_SCHEMA, 
                        temperature=0.0, 
                        service_tier="flex" 
                    ),
                )
                
                decision = json.loads(response.text)
                return {
                    "action": decision.get("action", "HOLD"),
                    "reasoning": decision.get("reasoning", "No reasoning provided.")
                }
                
            except Exception as e:
                error_msg = str(e)
                if "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg:
                    time.sleep(RETRY_DELAY_SECONDS) 
                else:
                    print(f"[ERROR] PortfolioManager execution failed: {error_msg}")
                    break 
                    
        return {"action": "HOLD", "reasoning": "Sub-routine API timeout."}