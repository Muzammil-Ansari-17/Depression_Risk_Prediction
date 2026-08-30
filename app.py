import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, time, timedelta




with open("mental_health_risk_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("mental_health_threshold.pkl", "rb") as file:
    threshold = pickle.load(file)

with open("mental_health_features.pkl", "rb") as file:
    features = pickle.load(file)




st.set_page_config(
    page_title="Depression Risk Predictor",
    page_icon="🧠",
    layout="centered"
)




st.title("Mental Health Risk Predictor")

st.write(
    "This tool estimates whether a person's screen-use and sleep pattern "
    "resembles the elevated-risk patterns found in the training data."
)




with st.expander("ℹ️ How to use this tool", expanded=False):

    st.markdown(
        """
### How the questionnaire works

The model uses information about **screen habits and sleep habits**.

Some questions use a **1–6 scale** because that is how the original
dataset recorded the responses.

**1 → lower end of the measured behavior**  
**6 → higher end of the measured behavior**

The supplied dataset does not include the original text labels for
every point on this 1–6 scale, so the app does not invent descriptions
such as *Never*, *Sometimes*, or *Always*.

---

### Screen-use questions

You will enter three screen-use ratings:

- Normal-day screen use
- Weekday screen use
- Weekend screen use

The app automatically calculates:

`Screen Time Index = average of these three ratings`

You do **not** need to calculate the index yourself.

---

### Sleep-quality questions

You will rate four types of sleep difficulty:

- Difficulty falling asleep
- Repeated awakening during sleep
- Disturbed sleep
- Early awakening

The app automatically calculates:

`Sleep Quality Index = average of the four ratings`

Higher values therefore represent a higher overall level of the
sleep-problem items in this dataset.

---

### Average sleep

Enter approximately how many hours you normally sleep per night.

---

### Weekend sleep midpoint

Instead of asking you for a confusing number such as `4.5`,
the app asks for your usual weekend **bedtime and wake-up time**.

For example:

**12:00 AM → 8:00 AM**

has a sleep midpoint of approximately:

**4:00 AM**

The app calculates this automatically.

---

### About the result

The percentage shown by the app is a **machine-learning risk estimate**.

It is **not a diagnosis of depression** and should not be interpreted
as one.
"""
    )




def calculate_sleep_midpoint(bed_time, wake_time):
    """
    Convert bedtime and wake-up time into midpoint-of-sleep
    expressed as decimal hours after midnight.
    """

    base_date = datetime(2026, 1, 1)

    bed = datetime.combine(base_date.date(), bed_time)
    wake = datetime.combine(base_date.date(), wake_time)

    # If waking occurs earlier on the clock, assume next day
    if wake <= bed:
        wake += timedelta(days=1)

    midpoint = bed + (wake - bed) / 2

    decimal_hour = (
        midpoint.hour
        + midpoint.minute / 60
        + midpoint.second / 3600
    )

    return decimal_hour




with st.form("prediction_form"):

    st.subheader("Basic Information")

    sex = st.selectbox(
        "Sex",
        ["Boy", "Girl"],
        help="These are the two categories present in the training dataset."
    )

    st.divider()



    st.subheader("Screen Habits")

    st.caption(
        "Use the same 1–6 rating scale used in the original dataset. "
        "1 is the lower end and 6 is the higher end."
    )

    screen_normal_day = st.select_slider(
        "Screen use on a normal day",
        options=[1, 2, 3, 4, 5, 6],
        value=3,
        help=(
            "Rate your normal-day screen use from 1 to 6. "
            "Higher numbers represent the higher end of the "
            "screen-use scale in the dataset."
        )
    )

    screen_weekday = st.select_slider(
        "Screen use on weekdays",
        options=[1, 2, 3, 4, 5, 6],
        value=3,
        help="Rate your typical weekday screen use from 1 to 6."
    )

    screen_weekend = st.select_slider(
        "Screen use on weekends",
        options=[1, 2, 3, 4, 5, 6],
        value=3,
        help="Rate your typical weekend screen use from 1 to 6."
    )




    st.subheader("Sleep Quality")

    st.caption(
        "These questions also use the dataset's original 1–6 scale."
    )

    fall_asleep = st.select_slider(
        "Difficulty falling asleep",
        options=[1, 2, 3, 4, 5, 6],
        value=2,
        help=(
            "1 is the lower end and 6 is the higher end of this "
            "sleep-difficulty item."
        )
    )

    repeated_awake = st.select_slider(
        "Repeated awakening during sleep",
        options=[1, 2, 3, 4, 5, 6],
        value=2,
        help="Rate this sleep item using the original dataset's 1–6 scale."
    )

    disturbed_sleep = st.select_slider(
        "Disturbed sleep",
        options=[1, 2, 3, 4, 5, 6],
        value=2,
        help="Rate this sleep item using the original dataset's 1–6 scale."
    )

    early_awake = st.select_slider(
        "Early awakening",
        options=[1, 2, 3, 4, 5, 6],
        value=2,
        help="Rate this sleep item using the original dataset's 1–6 scale."
    )



    st.subheader("Sleep Schedule")

    avg_sleep_hours = st.slider(
        "Average hours of sleep per night",
        min_value=1.0,
        max_value=12.0,
        value=8.0,
        step=0.25,
        help="Enter approximately how many hours you normally sleep per night."
    )

    bed_time = st.time_input(
        "Usual weekend bedtime",
        value=time(0, 0),
        help="For example, select 12:00 AM if you normally go to sleep at midnight."
    )

    wake_time = st.time_input(
        "Usual weekend wake-up time",
        value=time(8, 0),
        help="For example, select 8:00 AM if you normally wake around 8 AM."
    )




    submitted = st.form_submit_button(
        "Check Risk",
        use_container_width=True
    )




if submitted:



    screen_time_index = (
        screen_normal_day
        + screen_weekday
        + screen_weekend
    ) / 3


    sleep_quality_index = (
        fall_asleep
        + repeated_awake
        + disturbed_sleep
        + early_awake
    ) / 4


    midsleep_weekend_hours = calculate_sleep_midpoint(
        bed_time,
        wake_time
    )




    with st.expander("See calculated model inputs"):

        st.write(
            f"**Screen Time Index:** "
            f"{screen_time_index:.2f}"
        )

        st.write(
            f"**Sleep Quality Index:** "
            f"{sleep_quality_index:.2f}"
        )

        st.write(
            f"**Weekend Sleep Midpoint:** "
            f"{midsleep_weekend_hours:.2f} hours after midnight"
        )




    input_data = pd.DataFrame([{
        "sex": sex,
        "screen_time_index": screen_time_index,
        "screen_normal_day_1to6": screen_normal_day,
        "sleep_quality_index": sleep_quality_index,
        "avg_sleep_hours": avg_sleep_hours,
        "midsleep_weekend_hours": midsleep_weekend_hours,
        "sqi_fall_asleep_1to6": fall_asleep,
        "sqi_early_awake_1to6": early_awake
    }])




    input_data = input_data[features]



    probability = model.predict_proba(input_data)[0, 1]

    prediction = int(
        probability >= threshold
    )



    st.divider()

    st.subheader("Your Result")

    st.metric(
        "Estimated elevated-risk probability",
        f"{probability * 100:.1f}%"
    )

    if prediction == 1:
        st.warning("Higher-risk pattern detected")

        st.write(
            "This result is not a diagnosis. If you've also been experiencing "
            "persistent low mood, loss of interest, sleep problems, hopelessness, "
            "or difficulty functioning, consider speaking with a qualified "
            "mental-health professional."
        )

        st.subheader("What you can do next")

        st.markdown("""
        **Reliable resources**

        - [WHO — Depression: symptoms, treatment and self-care](https://www.who.int/news-room/fact-sheets/detail/depression)
        - [WHO — Depression overview](https://www.who.int/health-topics/depression)
        - [WHO — Psychological self-help resources](https://www.who.int/teams/mental-health-and-substance-use/treatment-care/Psychological-interventions/psychological-self-help-interventions)
        """)

        st.info(
            "If you're concerned about your mental health, consider talking to "
            "a psychologist, psychiatrist, counsellor, or another qualified "
            "healthcare professional."
        )


st.divider()

st.caption(
    "⚠️ This project is an educational machine-learning tool. "
    "It does not diagnose depression, replace a validated clinical "
    "screening instrument, or replace evaluation by a qualified "
    "health professional."
)