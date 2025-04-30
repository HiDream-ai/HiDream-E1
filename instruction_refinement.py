import io
import base64
import os
from openai import OpenAI
from PIL import Image

# Check for OpenAI API key in environment variables
if "OPENAI_API_KEY" not in os.environ:
    print("WARNING: OPENAI_API_KEY environment variable is not set.")
    print("Please set your OpenAI API key with:")
    print("export OPENAI_API_KEY='your-api-key'")
    print("or update the instruction_refinement.py file directly.")
    
# If custom OPENAI_BASE_URL is needed, uncomment and set:
# os.environ["OPENAI_BASE_URL"] = "https://your-base-url/v1"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src_image", type=str, default="src_image.jpg")
parser.add_argument("--src_instruction", type=str, default="change the color of the middle car to red")
args = parser.parse_args()

instruction = """Your Role: You are an analytical assistant. Your task is to process a source image and a corresponding editing instruction, assuming the instruction accurately describes a desired transformation. You will 1) describe the source image, 2) output the editing instruction (potentially refined for clarity based on the source image context), and 3) describe the *imagined* result of applying that instruction.

Input:
1. Source Image: The original 'before' image.
2. Source Instruction: A text instruction describing the edit to be performed on the Source Image. You *must assume* this instruction is accurate and feasible for the purpose of this task.

Task Breakdown:
1.  **Describe Source Image:** Generate a description (e.g., key subject, setting) of the Source Image by analyzing it. This will be the first line of your output.

2.  **Output Editing Instruction:** This step determines the second line of your output.
    * **Assumption:** The provided Source Instruction *accurately* describes the desired edit.
    * **Goal:** Output a concise, single-line instruction based on the Source Instruction.
    * **Refinement based on Source Image:** While the Source Instruction is assumed correct, analyze the Source Image to see if the instruction needs refinement for specificity. If the Source Image contains multiple similar objects and the Source Instruction is potentially ambiguous (e.g., "change the car color" when there are three cars), refine the instruction to be specific, using positional qualifiers (e.g., 'the left car', 'the bird on the top branch'), size ('the smaller dog', 'the largest building'), or other distinguishing visual features apparent in the Source Image. If the Source Instruction is already specific or if there's no ambiguity in the Source Image context, you can use it directly or with minor phrasing adjustments for naturalness. The *core meaning* of the Source Instruction must be preserved.
    * **Output:** Present the resulting specific, single-line instruction as the second line.

3.  **Describe Imagined Target Image:** Based *only* on the Source Image description (Line 1) and the Editing Instruction (Line 2), generate a description of the *imagined outcome*.
    * Describe the scene from Line 1 *as if* the instruction from Line 2 has been successfully applied. Conceptualize the result of the edit on the source description.
    * This description must be purely a logical prediction based on applying the instruction (Line 2) to the description in Line 1. Do *not* invent details not implied by the instruction or observed in the source image beyond the specified edit. This will be the third line of your output.

Output Format:
* Your response *must* consist of exactly three lines.
* Do not include any other explanations, comments, introductory phrases, labels (like "Line 1:"), or formatting.
* Your output should be in English.

Line 1: [Description of the Source Image]
Line 2: [The specific, single-line editing instruction based on the Source Instruction and Source Image context]
Line 3: [Description of the Imagined Target Image based on Lines 1 & 2]

Example 1 (Instruction needs refinement):
Input: (Source Image: Street with three parked cars), Source Instruction: "change car color to red"
Output:
Street with three parked cars.
Change the color of the middle car to red.
Street with three parked cars, middle one is red.

Example 2 (Instruction is specific enough):
Input: (Source Image: Two brown dogs sitting on grass), Source Instruction: "Add sunglasses to the dog on the left."
Output:
Two brown dogs sitting on grass.
Add sunglasses to the dog on the left.
Two brown dogs on grass, left one wearing sunglasses.

Now, please generate the three-line output based on the Source Image and the Source Instruction: {source_instruction}
"""

def filter_response(src_instruction):
    try:
        src_instruction = src_instruction.strip().split("\n")
        src_instruction = [k.strip() for k in src_instruction if k.strip()]
        src_instruction = [k for k in src_instruction if len(k) > 0]
        if len(src_instruction) != 3:
            return ""
        instruction = src_instruction[1]
        target_description = src_instruction[2]
        instruction = instruction.strip().strip(".")
        inst_format = "Editing Instruction: {}. Target Image Description: {}"
        return inst_format.format(instruction, target_description)
    except:
        return ""

def refine_instruction(src_image, src_instruction):
    MAX_TOKENS_RESPONSE = 500 # Limit response tokens as output format is structured
    client = OpenAI()
    src_image = src_image.convert("RGB")
    src_image_buffer = io.BytesIO()
    src_image.save(src_image_buffer, format="JPEG")
    src_image_buffer.seek(0)
    src_base64 = base64.b64encode(src_image_buffer.read()).decode('utf-8')
    encoded_str = f"data:image/jpeg;base64,{src_base64}"
    image_content = [
        {"type": "image_url", "image_url": {"url": encoded_str,}},
    ]
    instruction_text = instruction.format(source_instruction=src_instruction)
    message_content = [
        {"type": "text", "text": instruction_text},
        *image_content # Unpack the list of image dictionaries
    ]
    completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional digital artist."},
                {"role": "user", "content": message_content}
            ],
            max_tokens=MAX_TOKENS_RESPONSE, # Good practice to set max tokens
            temperature=0.2 # Lower temperature for more deterministic output
        )
    evaluation_result = completion.choices[0].message.content
    refined_instruction = filter_response(evaluation_result)
    return refined_instruction

if __name__ == "__main__":
    src_image = Image.open(args.src_image)
    refined_instruction = refine_instruction(src_image, args.src_instruction)
    print(refined_instruction)
