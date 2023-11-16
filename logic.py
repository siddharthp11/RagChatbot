
import openai as ai

class ChatBot:
    def __init__(self, retrieval, model='curie', ):
        self.model = model
        self.retrieval = retrieval 

    def can_initialize(): 
        try:
            ai.Model.list()
        except ai.error.AuthenticationError as e:
            return False
        else:
            return True

    def get_response(self, question):
        contexts = self.retrieval(question)
        context_as_string = ' '.join(contexts)
        prompt = """Answer the question as truthfully as possible using the provided text, if the answer is not contained in the text, say ' I don't know'."

        Context: 
        {ctx}

        Q: {question}
        A:""".format(ctx=context_as_string, question=question)

        try: 
            response = ai.Completion.create(
                prompt=prompt,
                temperature=0,
                max_tokens=300,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                model="text-davinci-002"
            )["choices"][0]["text"].strip(" \n")

            return contexts, response
        except Exception as e: 
            return 'No context found', 'No response'

