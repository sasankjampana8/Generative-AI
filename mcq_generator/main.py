from langchain_ollama import OllamaLLM
import json

class MCQGenerator:
    def __init__(self) -> None:
        self.system_prompt = """
        You are an expert MCQ generator. You will be given a topic on which you are 
        to create questions with multiple-choice options in the following format:
        
        {
            "Question": "Your generated question here",
            "Options": {
                "a": "Option A",
                "b": "Option B",
                "c": "Option C",
                "d": "Option D"
            }
        }
        Make sure each question has one correct answer and that all options are plausible.
        """
        
    def generate_content(self, topic):
        self.model = OllamaLLM(model='llama3')
        request = {"prompt": f"{self.system_prompt} \nTopic: {topic}"}
        response = self.model.invoke(str(request))
        return response
        

if __name__ == "__main__":
    mcq_generator = MCQGenerator()  # Create an instance of the class
    topic = "Artificial Intelligence"  # Example topic
    response = mcq_generator.generate_content(topic)  # Pass the topic as an argument
    print(json.dumps(response, indent=4))  # Pretty-print the response as JSON
