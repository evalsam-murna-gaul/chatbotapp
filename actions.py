from transformers import AutoModelForCausalLM, AutoTokenizer
from rasa_sdk import Action
from rasa_sdk.events import SlotSet

class ActionChatWithLlama(Action):
    def name(self):
        return "action_chat_with_llama"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get("text")

        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        inputs = tokenizer(user_message, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=100)
        bot_reply = tokenizer.decode(output[0], skip_special_tokens=True)

        dispatcher.utter_message(text=bot_reply)
        return []
