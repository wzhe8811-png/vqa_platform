import torch
from transformers import LlavaForConditionalGeneration, AutoModelForCausalLM, AutoProcessor, CLIPImageProcessor, LlamaTokenizer
try:
    from transformers import Qwen2Tokenizer
except ImportError:
    Qwen2Tokenizer = None
import os


class VisionToTextReasoner:
    def __init__(self, vision_model_path, llm_model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # --- 1. Load LLaVA ---
        print(f"Loading LLaVA: {vision_model_path}")
        try:
            self.vision_model = LlavaForConditionalGeneration.from_pretrained(
                vision_model_path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
            )
            print("Forcing Slow Tokenizer for LLaVA...")
            self.vision_image_processor = CLIPImageProcessor.from_pretrained(vision_model_path)
            self.vision_processor = AutoProcessor.from_pretrained(vision_model_path)
            self.vision_tokenizer = LlamaTokenizer.from_pretrained(vision_model_path, use_fast=False, legacy=False)
            if self.vision_tokenizer.pad_token is None:
                self.vision_tokenizer.pad_token = self.vision_tokenizer.eos_token
            self.use_manual_processing = True
            print("✅ LLaVA Loaded")
        except Exception as e:
            print(f"❌ LLaVA Load Failed: {e}")
            raise e

        # --- 2. Load Qwen ---
        print(f"Loading Qwen: {llm_model_path}")
        try:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True
            )
            
            if Qwen2Tokenizer is not None:
                print("Using explicit Qwen2Tokenizer...")
                self.llm_tokenizer = Qwen2Tokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
            else:
                print("Falling back to AutoTokenizer...")
                self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
            
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            print("✅ Qwen Loaded")
        except Exception as e:
            print(f"❌ Qwen Load Failed: {e}")
            raise e

    def get_visual_description(self, image_path, question_hint=""):
        from PIL import Image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert('RGB')
        prompt_text = "<image>\nUSER: 请用中文详细描述这张图片。\nASSISTANT:"
        
        inputs = self.vision_processor(text=prompt_text, images=image, return_tensors='pt').to(
            self.device, dtype=torch.float16
        )
        
        with torch.no_grad():
            output_ids = self.vision_model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.0, pad_token_id=self.vision_tokenizer.pad_token_id)
        
        generated_text = self.vision_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "ASSISTANT:" in generated_text:
            return generated_text.split("ASSISTANT:")[-1].strip()
        return generated_text.strip()

    def get_llm_caller(self):
        def llm_caller(prompt, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=512):
            messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
            
            # Handle chat template safely
            try:
                if hasattr(self.llm_tokenizer, 'apply_chat_template'):
                    text = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    text = "\\n\\nSystem:\\nYou are a helpful assistant.\\n\\nUser:\\n{prompt}\\n\\nAssistant:\\n".format(prompt=prompt)
            except:
                # Fallback for any issues with apply_chat_template
                text = "\\n\\nSystem:\\nYou are a helpful assistant.\\n\\nUser:\\n{prompt}\\n\\nAssistant:\\n".format(prompt=prompt)
                
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": self.llm_tokenizer.pad_token_id,
                }
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = top_p
                generated_ids = self.llm_model.generate(**model_inputs, **gen_kwargs)
            
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            return self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
        return llm_caller
