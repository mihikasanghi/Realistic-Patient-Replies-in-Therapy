import dspy
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
import random

@dataclass
class PatientPersona:
    name: str
    age: int
    occupation: str
    background: str
    personality_traits: List[str]
    mental_health_history: str

@dataclass
class ConversationContext:
    session_number: int
    therapy_approach: str
    current_topic: str
    previous_patient_statement: str
    
    
def patient_persona_to_string(persona: PatientPersona) -> str:
    return f"""
    Name: {persona.name}
    Age: {persona.age}
    Occupation: {persona.occupation}
    Background: {persona.background}
    Personality Traits: {', '.join(persona.personality_traits)}
    Mental Health History: {persona.mental_health_history}
    """

def conversation_context_to_string(context: ConversationContext) -> str:
    return f"""
    Session Number: {context.session_number}
    Therapy Approach: {context.therapy_approach}
    Current Topic: {context.current_topic}
    Previous Patient Statement: {context.previous_patient_statement}
    """

class PersonaMoodEvaluator(dspy.Signature):
    """Evaluate the realism and consistency of a given persona-mood combination in a therapy context.

    Guidelines for Evaluation:
    1. Consistency: Ensure the mood aligns with the patient's background and mental health history.
    2. Contextual Appropriateness: Consider if the mood is fitting for the current therapy session and topic.
    3. Complexity: Avoid overly simplistic mood-persona combinations; real patients often have complex emotional states.
    4. Temporal Factors: Take into account recent events or progress in therapy that might influence mood.
    5. Personality Influence: Consider how the patient's personality traits might affect their mood expression.

    Scoring:
    - Score on a scale of 0.0 to 1.0, where 1.0 is perfectly realistic and consistent.
    - Scores below 0.5 indicate significant inconsistencies or unrealistic combinations.
    - Aim for scores above 0.7 for acceptable realism.

    Output Format:
    1. Provide a numerical score.
    2. Offer a brief explanation for the score, highlighting strengths or concerns.
    3. If applicable, suggest minor adjustments to improve realism.
    """
    persona: str = dspy.InputField()
    mood: str = dspy.InputField()
    context: str = dspy.InputField()

    realism_score: float = dspy.OutputField(desc="A float between 0.0 and 1.0")
    explanation: str = dspy.OutputField()
    suggested_adjustments: Optional[str] = dspy.OutputField()
    
    
class PatientReplyGenerator(dspy.Signature):
    """Generate a realistic and contextually appropriate patient reply in a therapy session.

    Guidelines for Generation:
    1. Maintain Consistency: Ensure the reply aligns with the patient's persona, mood, and previous statements.
    2. Reflect Therapeutic Progress: Consider the session number and any progress made in previous sessions.
    3. Incorporate Resistance or Openness: Based on the patient's personality and current mood, include appropriate levels of resistance or openness to therapy.
    4. Use Appropriate Language: Match the patient's educational background and typical speech patterns.
    5. Include Non-verbal Cues: Incorporate descriptions of tone, pauses, or physical gestures to enrich the response.
    6. Address the Therapist's Statement: Ensure the reply is a relevant response to the therapist's last statement or question.
    7. Reflect Internal Struggle: If appropriate, showcase the patient's internal conflicts or mixed feelings about their issues or the therapy process.

    Output Format:
    1. Start the reply with "Patient:" followed by the verbal response.
    2. Include non-verbal cues or actions in parentheses.
    3. Keep the reply length appropriate - typically 1-3 sentences, unless a longer response is contextually justified.
    """
    persona: str = dspy.InputField()
    mood: str = dspy.InputField()
    context: str = dspy.InputField()
    therapist_statement: str = dspy.InputField()

    patient_reply: str = dspy.OutputField()


class ReplyEvaluator(dspy.Signature):
    """Evaluate the realism and appropriateness of a generated patient reply in a therapy context.

    Evaluation Criteria:
    1. Consistency: Does the reply align with the patient's persona, mood, and conversation context?
    2. Relevance: Is the reply a suitable response to the therapist's statement?
    3. Emotional Congruence: Does the emotional tone of the reply match the patient's current mood and situation?
    4. Authenticity: Does the reply sound natural and not overly scripted?
    5. Therapeutic Engagement: Does the reply show an appropriate level of engagement with the therapeutic process?
    6. Language and Articulation: Is the language use consistent with the patient's background and emotional state?
    7. Depth: Does the reply demonstrate appropriate depth given the context and patient's characteristics?

    Scoring:
    - Score on a scale of 0.0 to 1.0, where 1.0 is a perfectly realistic and appropriate reply.
    - Scores below 0.5 indicate significant issues with the reply.
    - Aim for scores above 0.7 for acceptable realism.

    Output Format:
    1. Provide a numerical score.
    2. Offer a brief explanation for the score, highlighting strengths and areas for improvement.
    3. If the score is low, provide specific suggestions for improving the reply.
    """

    persona: str = dspy.InputField()
    mood: str = dspy.InputField()
    context: str = dspy.InputField()
    therapist_statement: str = dspy.InputField()
    patient_reply: str = dspy.InputField()

    realism_score: float = dspy.OutputField()
    explanation: str = dspy.OutputField()
    improvement_suggestions: Optional[str] = dspy.OutputField()

class PatientReplyWorkflow:
    def __init__(self, openai_api_key: str, realism_threshold: float = 0.7, max_attempts: int = 3):
        self.openai_api_key = openai_api_key
        self.realism_threshold = realism_threshold
        self.max_attempts = max_attempts
        self.lm = dspy.OpenAI(
            api_key=openai_api_key,
            model="gpt-4"  # Explicitly specify GPT-4 model
        )
        dspy.settings.configure(lm=self.lm)

    # Rest of the class implementation remains the same
    def generate_patient_reply(self,
                               persona: PatientPersona,
                               mood: str,
                               context: ConversationContext,
                               therapist_statement: str) -> Tuple[str, float, str]:
        # Convert objects to string representations
        persona_str = patient_persona_to_string(persona)
        context_str = conversation_context_to_string(context)

        # Evaluate Persona-Mood Combination
        mood_evaluator = dspy.Predict(PersonaMoodEvaluator)
        mood_eval_result = mood_evaluator(persona=persona_str, mood=mood, context=context_str)

        # Ensure realism_score is a float
        try:
            realism_score = float(mood_eval_result.realism_score)
        except ValueError:
            print(f"Warning: Invalid realism score: {mood_eval_result.realism_score}")
            realism_score = 0.0

        if realism_score < self.realism_threshold:
            print(f"Warning: Persona-Mood combination may not be realistic. Score: {realism_score}")
            print(f"Explanation: {mood_eval_result.explanation}")
            if mood_eval_result.suggested_adjustments:
                print(f"Suggested adjustments: {mood_eval_result.suggested_adjustments}")

        # Generate Patient Reply
        reply_generator = dspy.Predict(PatientReplyGenerator)
        reply_evaluator = dspy.Predict(ReplyEvaluator)

        for attempt in range(self.max_attempts):
            generated_reply = reply_generator(
                persona=persona_str,
                mood=mood,
                context=context_str,
                therapist_statement=therapist_statement
            )

            # Evaluate Reply
            eval_result = reply_evaluator(
                persona=persona_str,
                mood=mood,
                context=context_str,
                therapist_statement=therapist_statement,
                patient_reply=generated_reply.patient_reply
            )

            # Ensure realism_score is a float
            try:
                eval_realism_score = float(eval_result.realism_score)
            except ValueError:
                print(f"Warning: Invalid evaluation realism score: {eval_result.realism_score}")
                eval_realism_score = 0.0

            if eval_realism_score >= self.realism_threshold:
                return generated_reply.patient_reply, eval_realism_score, eval_result.explanation

            print(f"Attempt {attempt + 1}: Reply not realistic enough. Score: {eval_realism_score}")
            print(f"Explanation: {eval_result.explanation}")
            if eval_result.improvement_suggestions:
                print(f"Improvement suggestions: {eval_result.improvement_suggestions}")

        return generated_reply.patient_reply, eval_realism_score, eval_result.explanation

if __name__ == "__main__":
    
    openai_api_key = "your_api_key_here"
    
    workflow = PatientReplyWorkflow(openai_api_key)
    
    number_of_replies = 20
    
    personas = [
        PatientPersona(
            name="Alex",
            age=28,
            occupation="Software Developer",
            background="Recently divorced, struggling with work-life balance",
            personality_traits=["introverted", "analytical", "perfectionist"],
            mental_health_history="History of mild depression, first time in therapy"
        ),
        PatientPersona(
            name="Sarah",
            age=35,
            occupation="Elementary School Teacher",
            background="Single parent of two, dealing with burnout",
            personality_traits=["empathetic", "organized", "anxious"],
            mental_health_history="Diagnosed with generalized anxiety disorder, in therapy for 6 months"
        ),
        PatientPersona(
            name="Michael",
            age=42,
            occupation="Marketing Executive",
            background="Workaholic, recently passed over for promotion",
            personality_traits=["ambitious", "competitive", "impatient"],
            mental_health_history="No prior therapy, experiencing symptoms of burnout and insomnia"
        ),
        PatientPersona(
            name="Emily",
            age=19,
            occupation="College Student",
            background="First-generation college student, struggling with academic pressure",
            personality_traits=["creative", "sensitive", "self-critical"],
            mental_health_history="History of social anxiety, started therapy 3 months ago"
        ),
        PatientPersona(
            name="Robert",
            age=55,
            occupation="Construction Worker",
            background="Recovering alcoholic, trying to rebuild relationships with family",
            personality_traits=["stoic", "hardworking", "guarded"],
            mental_health_history="Completed rehab 1 year ago, in therapy for anger management"
        ),
        PatientPersona(
            name="Lisa",
            age=31,
            occupation="Nurse",
            background="Working in high-stress hospital environment, dealing with compassion fatigue",
            personality_traits=["compassionate", "dedicated", "perfectionist"],
            mental_health_history="Experiencing symptoms of PTSD, first time in therapy"
        ),
        PatientPersona(
            name="David",
            age=48,
            occupation="Small Business Owner",
            background="Recently filed for bankruptcy, marriage under strain",
            personality_traits=["risk-taker", "optimistic", "stubborn"],
            mental_health_history="History of mild depression, returning to therapy after 5 years"
        ),
        PatientPersona(
            name="Sophia",
            age=23,
            occupation="Aspiring Actor",
            background="Moved to big city to pursue dreams, feeling lonely and overwhelmed",
            personality_traits=["extroverted", "ambitious", "sensitive"],
            mental_health_history="No prior therapy, experiencing symptoms of depression and anxiety"
        ),
        PatientPersona(
            name="James",
            age=62,
            occupation="Recently Retired Engineer",
            background="Struggling to find purpose post-retirement, wife diagnosed with cancer",
            personality_traits=["logical", "reserved", "routine-oriented"],
            mental_health_history="No prior therapy, experiencing grief and adjustment issues"
        ),
        PatientPersona(
            name="Maria",
            age=37,
            occupation="Freelance Graphic Designer",
            background="First-generation immigrant, balancing cultural expectations with personal goals",
            personality_traits=["creative", "adaptable", "people-pleaser"],
            mental_health_history="In therapy for 2 years, working on self-esteem and assertiveness"
        )
    ]

    moods = [
        "anxious and slightly defensive",
        "depressed and withdrawn",
        "optimistic but cautious",
        "frustrated and impatient",
        "calm and reflective",
        "angry and confrontational",
        "sad but hopeful",
        "overwhelmed and scattered",
        "content yet uncertain",
        "fearful and avoidant",
        "excited and talkative",
        "guilty and remorseful",
        "numb and disconnected",
        "irritable and restless",
        "grateful but worried",
        "confused and seeking clarity",
        "determined and focused",
        "vulnerable and open",
        "skeptical and guarded",
        "energetic but nervous",
        "resigned and apathetic",
        "curious and engaged",
        "ashamed and self-critical",
        "relieved but exhausted",
        "motivated yet apprehensive",
        "pessimistic and cynical",
        "confident and assertive",
        "lonely and seeking connection",
        "nostalgic and melancholic",
        "amused and light-hearted"
    ]
    
    contexts = [
        ConversationContext(
            session_number=3,
            therapy_approach="Cognitive Behavioral Therapy",
            current_topic="Work-related stress",
            previous_patient_statement="I feel overwhelmed by my project deadlines."
        ),
        ConversationContext(
            session_number=1,
            therapy_approach="Psychodynamic Therapy",
            current_topic="Childhood experiences",
            previous_patient_statement="I've always felt like I wasn't good enough for my parents."
        ),
        ConversationContext(
            session_number=7,
            therapy_approach="Mindfulness-Based Stress Reduction",
            current_topic="Anxiety management",
            previous_patient_statement="I tried the breathing exercise, but my mind kept wandering."
        ),
        ConversationContext(
            session_number=5,
            therapy_approach="Solution-Focused Brief Therapy",
            current_topic="Relationship issues",
            previous_patient_statement="Things have been better with my partner since we started communicating more."
        ),
        ConversationContext(
            session_number=2,
            therapy_approach="Acceptance and Commitment Therapy",
            current_topic="Values and goals",
            previous_patient_statement="I'm not sure what I really want in life anymore."
        ),
        ConversationContext(
            session_number=10,
            therapy_approach="Cognitive Behavioral Therapy",
            current_topic="Depression management",
            previous_patient_statement="I've been able to challenge some of my negative thoughts, but it's still hard."
        ),
        ConversationContext(
            session_number=4,
            therapy_approach="Dialectical Behavior Therapy",
            current_topic="Emotional regulation",
            previous_patient_statement="I lashed out at my coworker again, but I felt guilty immediately after."
        ),
        ConversationContext(
            session_number=8,
            therapy_approach="Interpersonal Therapy",
            current_topic="Social support",
            previous_patient_statement="I've been trying to reach out to friends more, but it feels awkward."
        ),
        ConversationContext(
            session_number=6,
            therapy_approach="Existential Therapy",
            current_topic="Life meaning and purpose",
            previous_patient_statement="Sometimes I wonder if anything I do really matters."
        ),
        ConversationContext(
            session_number=12,
            therapy_approach="Narrative Therapy",
            current_topic="Reframing life story",
            previous_patient_statement="I'm starting to see how I've been telling myself a negative story about my capabilities."
        ),
        ConversationContext(
            session_number=3,
            therapy_approach="Gestalt Therapy",
            current_topic="Present-moment awareness",
            previous_patient_statement="I noticed I was clenching my fists when talking about my boss."
        ),
        ConversationContext(
            session_number=9,
            therapy_approach="Art Therapy",
            current_topic="Self-expression",
            previous_patient_statement="The painting I made last session really helped me understand my feelings better."
        ),
        ConversationContext(
            session_number=2,
            therapy_approach="Family Systems Therapy",
            current_topic="Family dynamics",
            previous_patient_statement="I realized I've been playing the peacekeeper role in my family for years."
        ),
        ConversationContext(
            session_number=5,
            therapy_approach="Positive Psychology",
            current_topic="Strengths and resilience",
            previous_patient_statement="I've been trying to focus on what I'm good at instead of my weaknesses."
        ),
        ConversationContext(
            session_number=7,
            therapy_approach="Trauma-Focused Cognitive Behavioral Therapy",
            current_topic="Coping with traumatic memories",
            previous_patient_statement="The nightmares are less frequent now, but they're still intense when they happen."
        )
    ]

    therapist_replies = [
        "Thank you for sharing that. Can you tell me more about how this experience has been affecting you?",
        
        "It sounds like you've been going through a lot. How have you been coping with these feelings?",
        
        "I'm curious about your perspective on this. What do you think might be underlying these experiences?",
        
        "Let's explore that further. How do you think this relates to what we've discussed in previous sessions?",
        
        "It's important to acknowledge those feelings. Have you noticed any patterns in when they tend to arise?",
        
        "You've shown a lot of resilience in dealing with this. What strategies have you found most helpful so far?",
        
        "I wonder if we could take a moment to reflect on how this situation aligns with your personal values and goals.",
        
        "That must be challenging to deal with. How would you like things to be different?",
        
        "It's interesting that you mention that. How do you think this connects to other areas of your life?",
        
        "I appreciate your openness. Can you walk me through what a typical day looks like for you when dealing with this?",
        
        "Let's take a step back for a moment. How do you think someone you admire might handle a similar situation?",
        
        "It sounds like this has been weighing on you. What would it look like to show yourself some compassion in this situation?",
        
        "I'm hearing a lot of important points. Which aspect of this do you think is most crucial for us to focus on right now?",
        
        "You've made some important observations. How do you think understanding this might help you move forward?",
        
        "It's clear you've given this a lot of thought. What do you think might be a small, manageable step you could take to address this?",
        
        "I'm noticing some themes in what you're sharing. How do these experiences compare to similar situations you've faced in the past?",
        
        "That sounds really challenging. If you could change one thing about this situation, what would it be?",
        
        "You've mentioned several different aspects of this issue. Which one feels most pressing or important to you right now?",
        
        "I can see how much this matters to you. What do you think success or progress would look like in this situation?",
        
        "It's important that we explore this further. How do you think these experiences have shaped your view of yourself or the world?"
    ]
        
    for i in range(number_of_replies):
        print("Datapoint: ", i+1, " out of ", number_of_replies, "\nLogs:\n")
        persona = personas[random.randint(0, len(personas) - 1)]
        mood = moods[random.randint(0, len(moods) - 1)]
        context = contexts[random.randint(0, len(contexts) - 1)]
        therapist_statement = therapist_replies[random.randint(0, len(therapist_replies) - 1)]

        patient_reply, realism_score, explanation = workflow.generate_patient_reply(
            persona, mood, context, therapist_statement
        )
        with open('patient_replies.csv', 'a') as f:
            f.write(f"{patient_reply},{realism_score},{explanation},{persona.personality_traits}\n")
            