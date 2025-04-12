import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import re
from PIL import Image
import math

def combine_images(image_paths, grid_size=(2, 2), save_path='combined.png'):
    """
    Combines multiple images into a single image grid.
    
    :param image_paths: list of image file paths
    :param grid_size: tuple (rows, cols)
    :param save_path: filename to save combined image
    """
    rows, cols = grid_size
    assert len(image_paths) <= rows * cols, "Grid too small for number of images"

    # Load images
    images = [Image.open(path) for path in image_paths]

    # Resize to same size (based on first image)
    width, height = images[0].size
    images = [img.resize((width, height)) for img in images]

    # Create canvas
    combined = Image.new('RGB', (cols * width, rows * height), color=(255, 255, 255))

    # Paste each image into grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * width
        y = row * height
        combined.paste(img, (x, y))

    # Save final image
    combined.save(save_path)
    print(f"Saved combined image as: {save_path}")


def create_model_plot(model, mp, names, saveLocation):
    x = np.arange(len(names)) * 1.5
    bar_width = 0.35

    # Get values from your data source
    yes_vals = [mp[(model, metric, 'Yes')] for metric in names]
    some_vals = [mp[(model, metric, 'To some extent')] for metric in names]
    no_vals = [mp[(model, metric, 'No')] for metric in names]

    # Plot grouped bars
    plt.bar(x - bar_width, yes_vals, width=bar_width, label='Yes', color='green')
    plt.bar(x, some_vals, width=bar_width, label='To some extent', color='orange')
    plt.bar(x + bar_width, no_vals, width=bar_width, label='No', color='red')

    yMax = 0
    for i in range(len(x)):
        yMax = max(yMax, yes_vals[i], some_vals[i], no_vals[i])

    for i in range(len(x)):
        plt.text(x[i] - bar_width, yes_vals[i] + yMax * 0.0075, str(yes_vals[i]), ha='center')
        plt.text(x[i], some_vals[i] + yMax * 0.0075, str(some_vals[i]), ha='center')
        plt.text(x[i] + bar_width, no_vals[i] + yMax * 0.0075, str(no_vals[i]), ha='center')

    # Add labels and legend
    plt.xlabel('Feedback Metric')
    plt.ylabel('Frequency')
    plt.title(f'{model} Feedback Metric Ratings')
    plt.xticks(x, names, rotation=15)
    plt.legend()

    plt.tight_layout()

    plt.savefig(saveLocation)
    plt.clf()


def create_metric_plot(metric, mp, names, saveLocation):
    x = np.arange(len(names)) * 1.5
    bar_width = 0.35

    # Get values from your data source
    yes_vals = [mp[(model, metric, 'Yes')] for model in names]
    some_vals = [mp[(model, metric, 'To some extent')] for model in names]
    no_vals = [mp[(model, metric, 'No')] for model in names]

    # Plot grouped bars
    plt.bar(x - bar_width, yes_vals, width=bar_width, label='Yes', color='green')
    plt.bar(x, some_vals, width=bar_width, label='To some extent', color='orange')
    plt.bar(x + bar_width, no_vals, width=bar_width, label='No', color='red')

    yMax = 0
    for i in range(len(x)):
        yMax = max(yMax, yes_vals[i], some_vals[i], no_vals[i])

    for i in range(len(x)):
        plt.text(x[i] - bar_width, yes_vals[i] + yMax * 0.0075, str(yes_vals[i]), ha='center')
        plt.text(x[i], some_vals[i] + yMax * 0.0075, str(some_vals[i]), ha='center')
        plt.text(x[i] + bar_width, no_vals[i] + yMax * 0.0075, str(no_vals[i]), ha='center')

    # Add labels and legend
    plt.xlabel('LLM Tutor Name')
    plt.ylabel('Frequency')
    plt.title(f'{metric} Feedback Metric Ratings')
    plt.xticks(x, names, rotation=15)
    plt.legend()

    plt.tight_layout()

    plt.savefig(saveLocation)
    plt.clf()




def extract_dialogue_pairs(text):
    # Find all entries that start with Tutor: or Student:
    entries = re.findall(r'(Tutor:|Student:)\s*(.*?)\s*(?=Tutor:|Student:|$)', text, re.DOTALL)
    
    # Group into tutor-student pairs
    pairs = []
    i = 0
    while i < len(entries):
        if entries[i][0] == "Tutor:":
            tutor_text = entries[i][1].strip()
            student_text = entries[i+1][1].strip() if i+1 < len(entries) and entries[i+1][0] == "Student:" else ""
            if i == 0:
                pairs.append({
                    "[TUTOR] [INITIAL QUESTION]": tutor_text,
                    "[STUDENT] [INITIAL ANSWER]": student_text
                })
            else:
                pairs.append({
                    "[TUTOR] [FOLLOW UP]": tutor_text,
                    "[STUDENT] [FOLLOW UP]": student_text
                })
            i += 2
        else:
            # Skip standalone student entry if any
            i += 1
    
    return pairs

def load_data(filepath):
    """
    Load and parse the MRBench dataset from a JSON file.
    Returns a pandas DataFrame with processed conversations and annotation counts.
    """
    # Load the JSON data
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ratings labeling
    ratings_labeling = {'Yes': 2, 'To some extent': 2, 'No': 0}
    model_labeling = {'Sonnet': 0,
        'Llama318B': 1,
        'Llama31405B': 2,
        'GPT4': 3,
        'Mistral': 4,
        'Expert': 5,
        'Gemini': 6,
        'Phi3': 7,
        'Novice': 8}

    # Process each conversation
    model_metrics = {}
    conversations = []
    for item in data:
        conv_id = item['conversation_id']
        conv_history = item['conversation_history']
        
        # Process annotations if they exist
        if 'tutor_responses' in item:
            for model, response in item['tutor_responses'].items():

                if 'annotation' in response:
                    for metric, rating in response['annotation'].items():
                        #annotation_counts[model][f"{metric}_{rating}"] += 1

                        key = (model, metric, rating)
                        if key not in model_metrics:
                            model_metrics[key] = 0
                        model_metrics[key] += 1

                label_info = response['annotation'].items() # not efficient, but works
                label_dict = {k: ratings_labeling.get(v, v) for k, v in label_info}

                # Adds conversation data
                conversations.append({ 
                    #'conversation_id': conv_id, # Not important
                    'conversation_history': extract_dialogue_pairs(conv_history),
                    'tutor_response': response['response'],
                    **dict(label_dict),
                    'LLM Tutor Name': model_labeling[model]
                })


    total_metric_ratings_freq = {}
    total_model_ratings_freq = {}
    model_set = set([])

    for key in model_metrics:
        new_key1 = (key[1], key[2]) # don't include the model
        new_key2 = (key[0], key[2]) # don't include the model
        model_set.add(key[0])
        #print(new_key)
        if new_key1 not in total_metric_ratings_freq:
            total_metric_ratings_freq[new_key1] = 0

        if new_key2 not in total_model_ratings_freq:
            total_model_ratings_freq[new_key2] = 0

        # Combine stats from all models into overall distribution
        total_metric_ratings_freq[new_key1] += model_metrics[key]
        total_model_ratings_freq[new_key2] += model_metrics[key]


    print('done')
    #print(len(metric_rating), len(metric_ratings_freq))

    ##########
    # Make Total Feedback Metric Plot
    ##########

    names = ['Mistake_Identification', 'Mistake_Location', 'Providing_Guidance', 'Actionability']
    x = np.arange(len(names)) * 1.5
    bar_width = 0.35

    # Get values from your data source
    yes_vals = [total_metric_ratings_freq[(metric, 'Yes')] for metric in names]
    some_vals = [total_metric_ratings_freq[(metric, 'To some extent')] for metric in names]
    no_vals = [total_metric_ratings_freq[(metric, 'No')] for metric in names]

    plt.figure(figsize=(6.4, 4.8))

    # Plot grouped bars
    plt.bar(x - bar_width, yes_vals, width=bar_width, label='Yes', color='green')
    plt.bar(x, some_vals, width=bar_width, label='To some extent', color='orange')
    plt.bar(x + bar_width, no_vals, width=bar_width, label='No', color='red')

    yMax = 0
    for i in range(len(x)):
        yMax = max(yMax, yes_vals[i], some_vals[i], no_vals[i])

    for i in range(len(x)):
        plt.text(x[i] - bar_width, yes_vals[i] + yMax * 0.0075, str(yes_vals[i]), ha='center')
        plt.text(x[i], some_vals[i] + yMax * 0.0075, str(some_vals[i]), ha='center')
        plt.text(x[i] + bar_width, no_vals[i] + yMax * 0.0075, str(no_vals[i]), ha='center')

    # Add labels and legend
    plt.xlabel('Feedback Metric')
    plt.ylabel('Frequency')
    plt.title('Total Feedback Metric Ratings')
    plt.xticks(x, names, rotation=15)
    plt.legend()

    plt.tight_layout()

    plt.savefig("figures/total_metric_ratings_plt.png")
    plt.clf()

    model_img_paths = []
    for model in model_set:
        model_img_paths.append(f"figures/models/{model}_metric_ratings_plt.png")
        create_model_plot(model, model_metrics, names, model_img_paths[-1])


    ##########
    # Make Total Feedback Model Plot
    ##########


    x = np.arange(len(model_set)) * 1.5
    bar_width = 0.35

    # Get values from your data source
    yes_vals = [total_model_ratings_freq[(model, 'Yes')] for model in model_set]
    some_vals = [total_model_ratings_freq[(model, 'To some extent')] for model in model_set]
    no_vals = [total_model_ratings_freq[(model, 'No')] for model in model_set]

    plt.figure(figsize=(12, 6))

    # Plot grouped bars
    plt.bar(x - bar_width, yes_vals, width=bar_width, label='Yes', color='green')
    plt.bar(x, some_vals, width=bar_width, label='To some extent', color='orange')
    plt.bar(x + bar_width, no_vals, width=bar_width, label='No', color='red')

    yMax = 0
    for i in range(len(x)):
        yMax = max(yMax, yes_vals[i], some_vals[i], no_vals[i])

    for i in range(len(x)):
        plt.text(x[i] - bar_width, yes_vals[i] + yMax * 0.0075, str(yes_vals[i]), ha='center')
        plt.text(x[i], some_vals[i] + yMax * 0.0075, str(some_vals[i]), ha='center')
        plt.text(x[i] + bar_width, no_vals[i] + yMax * 0.0075, str(no_vals[i]), ha='center')

    # Add labels and legend
    plt.xlabel('LLM Tutor Name')
    plt.ylabel('Frequency')
    plt.title('Feedback LLM Tutor Ratings')
    plt.xticks(x, model_set, rotation=15)
    plt.legend()

    plt.tight_layout()

    plt.savefig("figures/total_model_ratings_plt.png")
    plt.clf()


    
    metric_img_paths = []
    for name in names:
        metric_img_paths.append(f"figures/metrics/{name}_metric_ratings_plt.png")
        create_metric_plot(name, model_metrics, model_set, metric_img_paths[-1])

    num_models = len(model_img_paths)
    num_metrics = len(metric_img_paths)
    combine_images(model_img_paths,
                   grid_size=(int(num_models//np.sqrt(num_models)), int(np.ceil(num_models/np.sqrt(num_models)))),
                   save_path="figures/combined_models_plt.png")
    combine_images(metric_img_paths,
                   grid_size=(int(num_metrics//np.sqrt(num_metrics)), int(np.ceil(num_metrics/np.sqrt(num_metrics)))),
                   save_path="figures/combined_metrics_plt.png")
    

    # Convert to DataFrame
    df = pd.DataFrame(conversations)
    print(f"\nTotal LLM Tutor Responses: {len(df)}")
    print(f"Data should have (x) LLM Tutor Responses: {2480}")
    print(f"Data should have (x) number of Conversations: {300}")
    return df

def main():
    filepath = 'data/mrbench_v3_devset.json'
    
    df = load_data(filepath)

    # NOTE
    # Could include distribution of classes in model (get this from create_model_plot & create_metric_plot)
    # Could Use Math Embedding with a Concat of text
    # Could use Ensemble methods with other models
    # Could train on other Math problems
    # Could add the 4 unused y labels in predicting the active y label

    # Save to CSV
    df.to_csv('data/clean_parsed_conversations.csv', index=False)
    

if __name__ == "__main__":
    main()