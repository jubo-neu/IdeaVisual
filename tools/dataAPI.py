import os
import json
import dashscope
import pandas as pd


def read_image_paths_from_folder(folder_path):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)


def extract_id_from_filename(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def build_messages(image_path):
    system_prompt = """
    You will see an image that shows a specific perspective of an object (e.g., the back, top, front, etc.). Please describe the image strictly based on its details, avoiding any subjective guesses or excessive imagination, and ensuring that the description faithfully reflects the details displayed in the image. Follow these requirements:

    Color: Accurately describe the color distribution of the object, including the main color and specific color details of certain parts (e.g., ears, horns, tail, etc.).

    Shape and Structure: Provide a detailed explanation of the shape features, outline lines, and the connection between various parts.

    Details: Precisely capture the visible details in the image, including angles, textures, and the shape of specific parts (e.g., ears, horns, tail, etc.).

    Obstruction and Perspective Features: Point out which parts are visible from this perspective and which parts are obstructed or not visible.

    Overall Appearance: Describe the overall texture, material feel, and the visual effect the object presents.

    Style and Consistency: Ensure that the description remains clear and coherent, avoiding subjective guesses or details beyond what is shown in the image.

    The description should be a continuous paragraph without bullet points or lists. For example:

    The overall color is predominantly white, with a compact and full shape, and a natural transition between the head and body. There are two small, light yellow horns on the top of the head, symmetrically positioned, slightly tilted backward. The ears are located on either side of the head and are dark brown, slightly extending outward. From the back, the left cheek is slightly visible with a dark brown patch and a bit of pink detail. The two short arms hang naturally on both sides of the body, in dark brown. The tail is a small, dark brown one, slightly bent downward, with a rounded tip. The overall surface is smooth, without noticeable texture details, and the colors transition smoothly and evenly. From this angle, the eyes, nose, or front details are not visible, and the overall appearance presents a simple, smooth look with a unified and stable visual effect.
    """

    messages = [
        {
            "role": "user",
            "content": [{"image": f"file://{image_path}"}] + [{"text": system_prompt}]
        }
    ]
    return messages


def call_api(messages):
    response = dashscope.MultiModalConversation.call(
        api_key="your key...",
        model='your model...',
        messages=messages
    )
    return response


def process_images_in_folder(folder_path):
    image_paths = read_image_paths_from_folder(folder_path)
    simplified_results = []

    for image_path in image_paths:
        try:
            image_id = extract_id_from_filename(image_path)
            messages = build_messages(image_path)
            result = call_api(messages)

            if result and result.status_code == 200:
                api_output = result.output
                choices = api_output.get('choices', [{}])
                message_content = choices[0].get('message', {}).get('content', [])

                text = "No description generated"
                if message_content and isinstance(message_content, list) and len(message_content) > 0:
                    first_element = message_content[0]
                    if isinstance(first_element, dict) and "text" in first_element:
                        text = first_element["text"]

                simplified_result = {
                    "filename": os.path.basename(image_path),
                    "id": image_id,
                    "text": text
                }
                simplified_results.append(simplified_result)
                print(f"Processed image: {image_path} with ID: {image_id}")
            else:
                error_msg = getattr(result, 'message', 'Unknown error') if result else 'No response from API'
                print(f"Error processing image {image_path}: {error_msg}")

        except Exception as e:
            print(f"Exception occurred while processing image {image_path}: {e}")

    if simplified_results:
        output_file = 'obj_describe_Eng_test.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=4)

        print(f"All simplified results saved to {output_file}")
    else:
        print("No valid results to save.")


if __name__ == "__main__":
    folder_path = '.../Answerer_dataset/test'
    process_images_in_folder(folder_path)
