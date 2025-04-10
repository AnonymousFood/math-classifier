import json
import pandas as pd
from collections import Counter

def load_data(filepath):
    """
    Load and parse the MRBench dataset from a JSON file.
    Returns a pandas DataFrame with processed conversations and annotation counts.
    """
    # Load the JSON data
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize counters for annotations (9 models total)
    annotation_counts = {
        'Sonnet': Counter(),
        'Llama318B': Counter(),
        'Llama31405B': Counter(),
        'GPT4': Counter(),
        'Mistral': Counter(),
        'Expert': Counter(),
        'Gemini': Counter(),
        'Phi3': Counter(),
        'Novice': Counter()
    }
    
    # Process each conversation
    conversations = []
    for item in data:
        conv_id = item['conversation_id']
        conv_history = item['conversation_history']
        
        # Process annotations if they exist
        if 'tutor_responses' in item:
            for model, response in item['tutor_responses'].items():
                if 'annotation' in response:
                    for metric, value in response['annotation'].items():
                        annotation_counts[model][f"{metric}_{value}"] += 1
        
        # Add conversation data
        conversations.append({
            'conversation_id': conv_id,
            'conversation_history': conv_history
        })

    # Print annotation statistics
    print("\nAnnotation Statistics:")
    print("=" * 50)

    # TODO plot out data distribution here somewhere
    for model in annotation_counts:
        print(f"\n{model} Model Results:")
        print("-" * 30)
        for key, count in annotation_counts[model].items():
            print(f"{key}: {count}")
    
    # Convert to DataFrame
    df = pd.DataFrame(conversations)
    return df

def main():
    filepath = 'mrbench_v3_devset.json'
    
    try:
        df = load_data(filepath)
        print(f"\nTotal conversations: {len(df)}")
        
        # NOTE Josh - could use [TUTOR] & [STUDENT] Tokens
        # Could Use Math Embedding with a Concat of text
        # Could use Ensemble methods with other models
        # Could train on other Math problems
        #

        # TODO Organize data
        # Conversation Id
        # xxxxxxx Conversation History
        # Tutor text
        # Student text
        # Model Name
        # Model Response

        # Mistake Identification (y1)
        # Mistake Location (y2)
        # Mistake Guidance (y3)
        # Actionability (y4)

        # TODO add the 4 labels to the conversation

        # Save to CSV
        #df.to_csv('parsed_conversations.csv', index=False)
        #print("\nData saved to parsed_conversations.csv")
        
    except FileNotFoundError:
        print(f"Error: Could not find file {filepath}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()