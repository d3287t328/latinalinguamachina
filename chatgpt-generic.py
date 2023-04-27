import os
import openai
# Set up the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a function to print text in color
def print_color(text, color):
    color_dict = {"blue": "\033[34m", "red": "\033[31m", "green": "\033[32m", "reset": "\033[0m"}
    print(color_dict[color] + text + color_dict["reset"])

# Main loop
while True:
    # Get user input and exit if the user types "exit"
    question = input("\033[34mQuid vis Domine?\n\033[0m")
    if question.lower() == "exit":
        print_color("Goodbye!", "red")
        break

    try:
        # Make the API call and process the response
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                    "insert your prompt here. if using multi line use triple quotes."},
                {"role": "user", "content": question}
            ]
        )

        # Print the response in green
        print_color(completion.choices[0].message.content, "green")

    except Exception as e:
        # Print errors in red
        print_color(f"Error: {e}", "red")

