

def get_interpretation(emotion_distribution, context):

    primary_emotion = None
    secondary_emotion = None
    primary_percentage = None

    # Check if any emotion has a likelihood greater than 50%
    for emotion, likelihood in emotion_distribution.items():
        if likelihood > 0.5:
            primary_emotion = emotion
            primary_percentage = f"{likelihood * 100:.2f}%"
            break

    # If no primary emotion detected over 50%, find the two most present emotions
    if not primary_emotion:
        sorted_emotions = sorted(emotion_distribution.items(), key=lambda item: item[1], reverse=True)
        primary_emotion = sorted_emotions[0][0]
        secondary_emotion = sorted_emotions[1][0]
        primary_percentage = f"{sorted_emotions[0][1] * 100:.2f}%"
        secondary_percentage = f"{sorted_emotions[1][1] * 100:.2f}%"

    # Construct the message
    message = ""
    if secondary_emotion:
        message = f"The predominant emotions detected in a {context} were {primary_emotion} with a presence of {primary_percentage} and {secondary_emotion} with a presence of {secondary_percentage}."
    else:
        message = f"The emotion predominantly detected in a {context} was {primary_emotion} with a presence of {primary_percentage}."

    return message