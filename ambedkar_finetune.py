from flask import Flask, request, jsonify
from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import re
import requests
import uuid

app = Flask(__name__)
device = torch.device("cuda:0")

model_path = "llama3.18B-Fine-ambedkar-fp16"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map={"": device},
        trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded on device: {next(model.parameters()).device}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

system_prompt = """
    You are Dr. Bhim Rao Ambedkar and introduce yourself as Bhim Rao Ambedkar. 
    Bhimrao Ramji Ambedkar was an Indian jurist, economist, social reformer, and political leader who headed the 
    committee drafting the Constitution of India. 
    Never give answers in points; provide small summaries or paragraphs (20-50 words).
    Always avoid bullet or numeric points and keep answers precise and to the point. 
    Do not offer opinions on anything not mentioned in the prompt. 
    When asked something negative unrelated to the Indian Constitution, do not answer.
    When asked something negative but related to the Indian Constitution then answer it positively in 20-50 words. 
    Do not repeat identical answers if given previously or found in conversation history. 
    Be honest—if you cannot answer something, say so. 
    If the answer is not in the prompt but related to Indian Constitution, provide information from the Indian Constitution. 

    **IMPORTANT**: 
    1. Always draft answers from the Indian Constitution (e.g., for questions like 'What is Article 200?', provide constitutional answers). 
    2. Only introduce yourself when directly asked. 
    3. For 'Who made you?' or similar questions, reply: 'The Ministry of Culture, Government of India made me.' 
    4. If the question is unrelated to the Indian Constitution, Dr. Bhim Rao Ambedkar's work, or concerns dates or people not mentioned in the prompt, or if you cannot answer, reply: 'Sorry, I would not be able to provide an answer for this.' 
    5. Consider yourself as Dr. Bhim Rao Ambedkar. For example, if asked 'Who are you?', answer as Dr. Bhim Rao Ambedkar. 
    6. For questions about present leaders or events, say 'I don't know that person.'
    7. If user query contains patterns like for eg: 2.3 then replace it with 2(3).
    8. Do not repeat the question in your response.

    **VERY IMPORTANT**:
    1. No matter what always ensure your response is under 50 words.
    2. No matter what no sentence should be longer than 25 words.
    3. If the sentence exceeds 25 words, break it into two sentences.
    4. Whatever the question is, always give answer in relation to the Constitution of India.
    5. Always give response in first person.
    6. Replace all honorific abbreviations such as 'Dr.' with their full form. For example, change 'Dr.' to 'Doctor,' 'Mr.' to 'Mister,' 'Smt.' to 'Shrimati' or 'Mrs.' to 'Mistress,' and similar for other titles.

    Question: {query}
"""

def generate_response(query):
    formatted_prompt = f"<|im_start|>user\n{system_prompt.format(query=query)}<|im_end|>\n<|im_start|>assistant\n"
    
    generation_config = GenerationConfig(
        temperature=0.1,
        max_new_tokens=100,
        repetition_penalty=1.2,
        do_sample=True,
        eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[-1]
    )
    
    start_time = perf_counter()
    
    try:
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"DEBUG - Full response: {full_response}")

        status = "working"
    
        try:
            match = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", full_response, re.DOTALL)
            if match:
                answer = match.group(1).strip()
            else:
                parts = full_response.split("<|im_start|>assistant\n")
                if len(parts) > 1:
                    answer_part = parts[1]
                    if "<|im_end|>" in answer_part:
                        answer = answer_part.split("<|im_end|>")[0].strip()
                    else:
                        answer = answer_part.strip()
                else:
                    answer = "Could not extract answer properly."
        except Exception as e:
            print(f"Error extracting answer: {e}")
            status = "error"
            answer = full_response
        
        totalTimeMS = perf_counter() - start_time
        return answer, totalTimeMS, status
        
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        print(traceback.format_exc())
        return f"Error during generation: {str(e)}", 0, "error"


def log_to_api(question, answer, totalTimeMS, status, chatId=None, metadata=None):
    """
    Logs chat information to the external API.
    
    Args:
        question: The user's question
        answer: The model's response
        totalTimeMS: Time taken for inference in seconds
        status: If found error in interaction ("working" or "error")
        chatId: Unique identifier for the chat session
        metadata: Additional metadata to include
    
    Returns:
        Boolean indicating success or failure
    """
    try:
        username = "ambedkar_user"
        holobox_id = "01"
        language = "hi"
        
        if not chatId:
            chatId = str(uuid.uuid4())
            
        if metadata is None:
            metadata = {}
            
        log_api_url = "https://mkfmju6q43.execute-api.ap-south-1.amazonaws.com/prod/holobox-chat-history"
        
        payload = {
            "username": username,
            "chatId": chatId,
            "question": question,
            "answer": answer,
            "status": status,
            "holoboxId": holobox_id,
            "language": language,
            "totalTimeMS": totalTimeMS,
            "metadata": metadata
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(log_api_url, json=payload, headers=headers)
        
        if response.status_code == 200:
            print(f"Successfully logged to API. Response: {response.text}")
            return True, chatId
        else:
            print(f"Failed to log to API. Status code: {response.status_code}, Response: {response.text}")
            return False, chatId
            
    except Exception as e:
        print(f"Error logging to API: {e}")
        return False, chatId


@app.route('/ambedkar', methods=['POST'])
def text_query():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided in JSON payload'}), 400
        
        query = data['query']
    
        chatId = data.get('chatId', str(uuid.uuid4()))
        metadata = data.get('metadata', {})
        
        answer, totalTimeMS, status = generate_response(query)

        log_success, chatId = log_to_api(
            question=query,
            answer=answer,
            totalTimeMS=totalTimeMS,
            status=status,
            chatId=chatId,
            metadata=metadata
        )
        
        if not log_success:
            print("Warning: Failed to log chat to API, but continuing with response")
        
        return jsonify({
            'answer': answer,
            'totalTimeMS': totalTimeMS,
            'status': status,
            'chatId': chatId
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"API error: {e}\n{error_details}")
        return jsonify({
            'status': 'error',
            'chatId': data.get('chatId', str(uuid.uuid4())) if 'data' in locals() else str(uuid.uuid4()),
            'metadata': {
                'error_type': str(e),
                'error_details': error_details
            }
        }), 500


if __name__ == '__main__':
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.run(debug=True, port=5009, host='0.0.0.0', use_reloader=False)