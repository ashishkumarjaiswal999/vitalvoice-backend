"""
medical_docs.py — Medical knowledge base for VitalVoice RAG pipeline.
Contains curated medical information from WHO, CDC, and MedlinePlus.
"""

MEDICAL_DOCUMENTS = [
    # ── Symptoms & Conditions ─────────────────────────────────────────────
    {
        "id": "fever_001",
        "category": "symptoms",
        "content": """Fever is a temporary increase in body temperature, often due to an illness.
        Normal body temperature is around 98.6°F (37°C). A fever is generally 100.4°F (38°C) or higher.
        Common causes: viral infections (flu, cold, COVID-19), bacterial infections, heat exhaustion.
        Warning signs requiring immediate care: fever above 103°F (39.4°C), fever lasting more than 3 days,
        severe headache, stiff neck, difficulty breathing, rash, confusion."""
    },
    {
        "id": "headache_001",
        "category": "symptoms",
        "content": """Headaches are one of the most common health complaints.
        Types: tension headache (most common, feels like pressure around head),
        migraine (severe throbbing, often one side, nausea, light sensitivity),
        cluster headache (severe pain around one eye),
        secondary headache (caused by another condition like hypertension or infection).
        Emergency signs: sudden severe headache ('thunderclap'), headache with fever and stiff neck,
        headache after head injury, headache with vision changes or confusion."""
    },
    {
        "id": "chest_pain_001",
        "category": "emergency",
        "content": """Chest pain can be a sign of a life-threatening emergency.
        Heart attack symptoms: pressure, squeezing, or tightness in chest, pain radiating to arm, jaw, or back,
        shortness of breath, sweating, nausea. CALL EMERGENCY SERVICES IMMEDIATELY.
        Other causes: angina, pulmonary embolism, pneumonia, acid reflux, muscle strain, panic attack.
        Rule: Always treat chest pain as a potential emergency until proven otherwise.
        First aid: Call emergency services, have patient rest, loosen tight clothing,
        if patient is not allergic, give aspirin 325mg."""
    },
    {
        "id": "diabetes_001",
        "category": "chronic_disease",
        "content": """Diabetes is a chronic condition affecting how the body processes blood sugar.
        Type 1: immune system attacks insulin-producing cells. Requires insulin therapy.
        Type 2: body doesn't use insulin properly. Most common type. Related to lifestyle factors.
        Gestational diabetes: occurs during pregnancy.
        Normal fasting blood glucose: 70-99 mg/dL.
        Prediabetes: 100-125 mg/dL fasting.
        Diabetes: 126 mg/dL or higher fasting on two separate tests.
        Symptoms: increased thirst, frequent urination, fatigue, blurred vision, slow healing wounds.
        Management: healthy diet, regular exercise, medication or insulin as prescribed."""
    },
    {
        "id": "hypertension_001",
        "category": "chronic_disease",
        "content": """High blood pressure (hypertension) is a common condition where blood force against artery walls is too high.
        Normal: less than 120/80 mmHg.
        Elevated: 120-129 systolic and less than 80 diastolic.
        Stage 1 hypertension: 130-139 systolic or 80-89 diastolic.
        Stage 2 hypertension: 140+ systolic or 90+ diastolic.
        Hypertensive crisis: 180+ systolic and/or 120+ diastolic — seek emergency care immediately.
        Risk factors: obesity, smoking, high salt diet, lack of exercise, stress, family history.
        Often has no symptoms — called the 'silent killer'."""
    },
    {
        "id": "malaria_001",
        "category": "infectious_disease",
        "content": """Malaria is a life-threatening disease caused by Plasmodium parasites transmitted by Anopheles mosquitoes.
        Common in tropical and subtropical regions including India, Africa, Southeast Asia.
        Symptoms: fever with chills and sweating (cyclical pattern every 48-72 hours),
        headache, muscle pain, fatigue, nausea, vomiting.
        Severe malaria: cerebral malaria (confusion, seizures), severe anemia, breathing difficulty.
        Diagnosis: blood smear test, rapid diagnostic test (RDT).
        Treatment: antimalarial medications (artemisinin-based combination therapy for P. falciparum).
        Prevention: insecticide-treated bed nets, indoor spraying, antimalarial prophylaxis when traveling."""
    },
    {
        "id": "dengue_001",
        "category": "infectious_disease",
        "content": """Dengue fever is a viral illness transmitted by Aedes mosquitoes.
        Common in India, Southeast Asia, Latin America, Caribbean.
        Symptoms: sudden high fever (104°F/40°C), severe headache, pain behind eyes,
        muscle and joint pain ('breakbone fever'), nausea, rash appearing 2-5 days after fever.
        Warning signs of severe dengue: abdominal pain, persistent vomiting, rapid breathing,
        bleeding gums, blood in urine or stool, fatigue, restlessness.
        No specific antiviral treatment. Management: rest, fluids, paracetamol for fever.
        AVOID aspirin and ibuprofen — can increase bleeding risk."""
    },
    {
        "id": "tuberculosis_001",
        "category": "infectious_disease",
        "content": """Tuberculosis (TB) is a bacterial infection caused by Mycobacterium tuberculosis.
        Primarily affects the lungs but can affect other organs.
        Symptoms: persistent cough lasting more than 3 weeks, coughing blood, chest pain,
        fatigue, weight loss, night sweats, fever.
        Spread through airborne droplets when infected person coughs, sneezes, or speaks.
        Latent TB: bacteria present but inactive, no symptoms, not contagious.
        Active TB: symptoms present, contagious.
        Treatment: 6-month course of multiple antibiotics. DOTS (Directly Observed Treatment) strategy.
        India has the highest TB burden globally. Free treatment available under RNTCP."""
    },
    {
        "id": "mental_health_anxiety_001",
        "category": "mental_health",
        "content": """Anxiety disorders are the most common mental health conditions globally.
        Types: Generalized Anxiety Disorder (GAD), panic disorder, social anxiety, phobias.
        Symptoms: excessive worry, restlessness, fatigue, difficulty concentrating,
        muscle tension, sleep problems, rapid heartbeat, sweating.
        Physical symptoms can mimic other conditions.
        Treatment: cognitive behavioral therapy (CBT), medication (SSRIs, SNRIs),
        lifestyle changes (exercise, sleep, reducing caffeine).
        Self-help: deep breathing exercises, mindfulness meditation, regular physical activity,
        limiting alcohol and caffeine, maintaining social connections."""
    },
    {
        "id": "mental_health_depression_001",
        "category": "mental_health",
        "content": """Depression is a common and serious medical illness affecting mood, thoughts, and behavior.
        Symptoms: persistent sadness or empty mood, loss of interest in activities,
        changes in appetite and weight, sleep disturbances, fatigue, feelings of worthlessness,
        difficulty concentrating, thoughts of death or suicide.
        Symptoms must persist for at least 2 weeks for diagnosis.
        Risk factors: family history, trauma, chronic illness, substance abuse, major life changes.
        Treatment: psychotherapy, antidepressant medications, lifestyle changes.
        Crisis: If someone expresses suicidal thoughts, seek immediate professional help.
        iCall India helpline: 9152987821. Vandrevala Foundation: 1860-2662-345."""
    },
    {
        "id": "first_aid_cpr_001",
        "category": "emergency",
        "content": """CPR (Cardiopulmonary Resuscitation) for adults:
        1. Check scene safety. Check if person is responsive — tap shoulder and shout.
        2. Call emergency services (112 in India) immediately.
        3. Check for breathing — look, listen, feel for no more than 10 seconds.
        4. Begin chest compressions: place heel of hand on center of chest,
           push hard and fast — at least 2 inches deep, 100-120 compressions per minute.
        5. Give rescue breaths (if trained): tilt head back, lift chin, pinch nose,
           give 2 breaths each lasting 1 second.
        6. Continue 30 compressions : 2 breaths ratio until help arrives.
        Hands-only CPR (no breaths): still effective for adults if uncomfortable with rescue breaths."""
    },
    {
        "id": "first_aid_burns_001",
        "category": "emergency",
        "content": """Burns first aid:
        Minor burns (small area, red, painful): cool with cool running water for 10-20 minutes.
        Do NOT use ice, butter, toothpaste, or any home remedies.
        Cover with clean non-stick bandage.
        Severe burns (large area, deep, white/charred): call emergency services immediately.
        Do NOT remove clothing stuck to burn. Do NOT break blisters.
        Keep patient warm to prevent shock.
        Chemical burns: remove contaminated clothing, flush with large amounts of water for 20 minutes.
        Electrical burns: do NOT touch patient until power source is off."""
    },
    {
        "id": "child_health_001",
        "category": "child_health",
        "content": """Child developmental milestones (WHO standards):
        0-3 months: smiles, tracks objects, responds to sounds.
        6 months: sits with support, babbles, recognizes familiar faces.
        12 months: stands with support, says 1-2 words, waves bye-bye.
        18 months: walks independently, says 10-20 words, points to objects.
        24 months: runs, speaks 2-word phrases, follows 2-step instructions.
        36 months: speaks in sentences, climbs stairs, plays with other children.
        Normal weight gain: 150-200g per week in first 3 months.
        Height: doubles by age 4, triples by age 13.
        Immunization schedule should follow national guidelines (India: BCG, OPV, DPT, Hepatitis B, MMR)."""
    },
    {
        "id": "nutrition_001",
        "category": "nutrition",
        "content": """Balanced diet guidelines (WHO/ICMR for India):
        Fruits and vegetables: at least 400g (5 portions) per day.
        Whole grains: rice, wheat, millets — should form 50-60% of calorie intake.
        Protein: dal, legumes, eggs, fish, lean meat — 0.8-1g per kg body weight.
        Dairy: 2-3 servings per day for calcium.
        Fats: limit saturated fats, avoid trans fats, prefer mustard/olive oil.
        Sugar: less than 10% of total energy intake.
        Salt: less than 5g per day.
        Water: 8-10 glasses (2-2.5 liters) per day.
        Iron-rich foods for anemia prevention: spinach, lentils, fortified foods.
        Vitamin D: sunlight exposure 15-20 minutes daily."""
    },
    {
        "id": "dental_health_001",
        "category": "dental",
        "content": """Common dental conditions:
        Tooth decay (cavities): caused by bacteria producing acid that damages enamel.
        Symptoms: toothache, sensitivity to hot/cold/sweet, visible holes.
        Treatment: fillings, root canal for severe decay, extraction if irreparable.
        Gum disease (gingivitis/periodontitis): red, swollen, bleeding gums.
        Caused by plaque buildup. Treatment: professional cleaning, improved oral hygiene.
        Tooth sensitivity: pain with hot/cold/sweet stimuli. Can indicate enamel erosion or gum recession.
        Oral hygiene: brush twice daily with fluoride toothpaste, floss daily,
        use mouthwash, replace toothbrush every 3 months.
        Visit dentist every 6 months for checkup and cleaning."""
    },
    {
        "id": "genetic_risk_001",
        "category": "genetics",
        "content": """Genetic health risks and family history:
        Heart disease: having a first-degree relative (parent, sibling) with heart disease
        before age 55 (men) or 65 (women) significantly increases your risk.
        Type 2 diabetes: 40% lifetime risk if one parent has diabetes, 70% if both parents have it.
        Breast cancer: BRCA1/BRCA2 gene mutations increase lifetime risk to 50-80%.
        Colorectal cancer: Lynch syndrome (hereditary) accounts for 3-5% of all colorectal cancers.
        Key preventive screenings based on family history:
        Cardiovascular: lipid panel, blood pressure monitoring from age 20.
        Diabetes: fasting glucose test annually if family history present.
        Cancer: earlier and more frequent screenings recommended.
        Genetic counseling recommended if multiple family members affected."""
    },
    {
        "id": "rural_health_001",
        "category": "rural_health",
        "content": """Common health issues in rural India and management:
        Waterborne diseases (diarrhea, typhoid, hepatitis A): ensure safe drinking water,
        use ORS for dehydration, boil water if unsure of safety.
        Vector-borne diseases (malaria, dengue, filariasis): use mosquito nets, eliminate standing water.
        Malnutrition: promote breastfeeding, give complementary foods from 6 months,
        use government nutrition programs (ICDS, NHM).
        Maternal health: antenatal care (4+ visits), institutional delivery, postnatal care.
        Ayushman Bharat: government health insurance scheme providing ₹5 lakh coverage per family.
        ASHA workers: first point of contact for health services in rural areas.
        Emergency referral: PHC → CHC → District Hospital → Medical College."""
    },
    {
        "id": "medicine_safety_001",
        "category": "medicine",
        "content": """Medicine safety guidelines:
        Always take medicines as prescribed — complete the full course.
        Common OTC medicines: Paracetamol (fever, pain) — max 4g/day for adults.
        Ibuprofen (pain, inflammation) — take with food, avoid if kidney problems.
        Antacids — for acidity, take 1-2 hours after meals.
        Drug interactions to avoid: Aspirin + blood thinners (increased bleeding risk).
        Paracetamol + alcohol (liver damage risk).
        Antibiotics: never take without prescription, complete full course,
        never share with others, never use leftover antibiotics.
        Store medicines: cool dry place, away from sunlight, out of reach of children.
        Check expiry dates before use."""
    }
]
