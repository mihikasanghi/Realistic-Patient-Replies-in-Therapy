# Realistic Patient Conversation Generator Using DSPy Framework

## Overview

This project implements a system for generating and evaluating realistic patient replies in simulated therapy conversations. It uses OpenAI's GPT-4 model through the `dspy` library to create dynamic and context-aware interactions between a virtual therapist and patient.

## Components

1. **Data Structures**
   - `PatientPersona`: Represents a patient's background information.
   - `ConversationContext`: Holds information about the current therapy session.

2. **Key Classes**
   - `PersonaMoodEvaluator`: Evaluates the realism of a patient's mood given their persona and context.
   - `PatientReplyGenerator`: Generates a patient's reply based on their persona, mood, context, and the therapist's statement.
   - `ReplyEvaluator`: Assesses the realism and appropriateness of the generated patient reply.
   - `PatientReplyWorkflow`: Orchestrates the entire process of generating and evaluating patient replies.

3. **Data Lists**
   - `personas`: A list of `PatientPersona` objects representing various patient backgrounds.
   - `moods`: A list of strings describing different emotional states.
   - `contexts`: A list of `ConversationContext` objects representing various therapy scenarios.
   - `therapist_replies`: A list of versatile therapist statements.

## How It Works

1. **Initialization**: The system is initialized with an OpenAI API key and configured to use the GPT-4 model.

2. **Workflow**:

   a. A random patient persona, mood, and conversation context are selected.

   b. The system evaluates if the persona-mood combination is realistic.

   c. A therapist statement is randomly selected or generated.

   d. The system generates a patient reply based on the persona, mood, context, and therapist statement.

   e. The generated reply is evaluated for realism and appropriateness.

   f. If the reply doesn't meet a certain realism threshold, the system attempts to generate a new reply (up to a maximum number of attempts).

3. **Output**: The system produces a patient reply along with a realism score and explanation. Personality traits have also been added on request.

## Usage

1. Ensure you have the required dependencies installed:
   ```
   pip install dspy openai
   ```

2. Set up your OpenAI API key:
   ```python
   openai_api_key = "your_api_key_here"
   ```

3. Initialize the workflow:
   ```python
   workflow = PatientReplyWorkflow(openai_api_key)
   ```

4. Generate a patient reply:
   ```python
   persona = random.choice(personas)
   mood = random.choice(moods)
   context = random.choice(contexts)
   therapist_statement = random.choice(therapist_replies)

   patient_reply, realism_score, explanation = workflow.generate_patient_reply(
       persona, mood, context, therapist_statement
   )
   ```

5. The generated reply, along with its realism score and explanation, can be used for further processing or output.
    If the realism score is not passed, an output like below is printed on the terminal (not included in CSV file):
    ```
    Warning: Persona-Mood combination may not be realistic. Score: 0.6
    Explanation: While it's possible for Robert to feel excited about starting therapy, his mood of being "excited and talkative" seems inconsistent with his personality traits of being stoic and guarded, especially considering this is the first session. His background as a recovering alcoholic and current therapy for anger management also suggest he might be more reserved or anxious. The topic of discussing childhood experiences, which might be a sensitive subject for him, also makes his excited mood seem less realistic.
    Suggested adjustments: A more realistic mood might be cautious optimism or apprehension. Robert might be hopeful about the potential benefits of therapy but also nervous about opening up and discussing painful topics.
    ```
    The multiple attempts it takes to generate a suitable reply is still however saved to the CSV file. This can be changed easily.

## Customization

- You can extend or modify the `personas`, `moods`, `contexts`, and `therapist_replies` lists to suit your specific needs.
- The output can include as many things as required (like the mood and context of the conversation).
- Adjust the `realism_threshold` and `max_attempts` parameters in the `PatientReplyWorkflow` initialization to control the quality and generation attempts of patient replies.


