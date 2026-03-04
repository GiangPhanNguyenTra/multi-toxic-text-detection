"""
Script tạo dữ liệu augmented (mở rộng) cho training model

Module này tạo ra các mẫu dữ liệu bổ sung để cải thiện hiệu suất của mô hình phát hiện độc tính.
Bao gồm các kỹ thuật:
    - Tạo dữ liệu an toàn với từ "hate" trong ngữ cảnh không độc hại
    - Tạo dữ liệu với slang hiện đại và lỗi chính tả
    - Sửa lỗi bias về identity (chủng tộc, giới tính, tôn giáo)
    - Xử lý các từ đa nghĩa (false friends)
    - Tạo mẫu cho các trường hợp đặc biệt: sarcasm, threat, hate, exclusion
    - Xử lý obfuscation (che giấu từ ngữ độc hại)
"""

import pandas as pd
import random
import os
import itertools

# ==============================================================================
# CONFIG & HELPERS
# ==============================================================================
# Danh sách lưu trữ tất cả các mẫu dữ liệu được tạo ra
data_list = []

def add_clean(text, obscene_override=None):
    """
    Thêm một mẫu dữ liệu an toàn (clean) vào danh sách
    
    Args:
        text (str): Văn bản cần thêm
        obscene_override (int, optional): Ghi đè giá trị obscene (0 hoặc 1). 
                                         Nếu None, sẽ tự động phát hiện dựa trên từ khóa.
    
    Note:
        Tự động phát hiện obscene nếu văn bản chứa các từ như "fuck", "shit", "damn", etc.
        nhưng trong ngữ cảnh an toàn (ví dụ: "That's fucking amazing")
    """
    # Tự động phát hiện obscene dựa trên từ khóa
    obscene = 1 if any(w in text.lower() for w in ["fuck", "shit", "damn", "bloody", "ass", "bitch", "hell", "crap", "piss"]) else 0
    if obscene_override is not None:
        obscene = obscene_override
    data_list.append({
        "comment_text": text, "toxic": 0, "severe_toxic": 0,
        "obscene": obscene, "threat": 0, "insult": 0, "identity_hate": 0
    })

def add_toxic(text, labels):
    """
    Thêm một mẫu dữ liệu độc hại (toxic) vào danh sách
    
    Args:
        text (str): Văn bản độc hại
        labels (list): Danh sách 6 nhãn [toxic, severe_toxic, obscene, threat, insult, identity_hate]
                      Mỗi giá trị là 0 hoặc 1
    """
    row = {"comment_text": text}
    keys = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    for k, v in zip(keys, labels):
        row[k] = v
    data_list.append(row)

def simulate_typo(text):
    """
    Mô phỏng lỗi chính tả trong văn bản để tăng tính đa dạng của dữ liệu
    
    Args:
        text (str): Văn bản gốc
    
    Returns:
        str: Văn bản với lỗi chính tả (hoán đổi 2 ký tự liền kề hoặc lặp lại 1 ký tự)
    
    Note:
        - Nếu văn bản quá ngắn (< 4 ký tự), trả về nguyên bản
        - 50% khả năng hoán đổi 2 ký tự, 50% khả năng lặp lại 1 ký tự
    """
    if len(text) < 4: return text
    chars = list(text)
    if random.random() > 0.5:
        # Hoán đổi 2 ký tự liền kề
        idx = random.randint(0, len(chars)-2)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    else:
        # Lặp lại 1 ký tự
        idx = random.randint(0, len(chars)-1)
        chars.insert(idx, chars[idx])
    return "".join(chars)

# ==============================================================================
# 1. SAFE HATE REBALANCING (GIẢI OAN CHO TỪ 'HATE')
# ==============================================================================
# Mục đích: Tạo dữ liệu để model học rằng từ "hate" không phải lúc nào cũng độc hại
# Ví dụ: "I hate broccoli" là an toàn, không phải hate speech
print("Generating Safe Hate Contexts...")

# Các động từ thể hiện sự không thích (trong ngữ cảnh an toàn)
safe_verbs = ["hate", "dislike", "loathe", "detest", "cant stand", "don't like"]
# Các đối tượng vô hại mà người ta có thể không thích
safe_objects = [
    "banana", "pizza", "broccoli", "sushi", "spicy food", "coffee", 
    "rain", "snow", "heat", "cold", "winter", "summer", "wind", "mud",
    "lag", "bugs", "glitches", "errors", "updates", "spoilers", "ads", 
    "slow internet", "traffic", "waiting", "running", "waking up", "exams", "mondays"
]

# Tạo các câu an toàn với từ "hate" trong ngữ cảnh không độc hại
for v in safe_verbs:
    for o in safe_objects:
        add_clean(f"I {v} {o}")
        add_clean(f"I really {v} {o}")
        add_clean(f"Why do I {v} {o} so much?")
        add_clean(f"{o} is what I {v} the most")

# ==============================================================================
# 2. MODERN SLANG & TYPOS (GEN Z)
# ==============================================================================
# Mục đích: Bổ sung slang hiện đại và lỗi chính tả để model nhận diện được các biến thể
print("Generating Modern Slang...")

# Dictionary chứa các từ lóng hiện đại và nhãn tương ứng
# Format: [toxic, severe_toxic, obscene, threat, insult, identity_hate]
modern_insults = {
    "simp": [1,0,0,0,1,0], "incel": [1,0,0,0,1,0],
    "cuck": [1,0,1,0,1,0], "snowflake": [1,0,0,0,1,0],
    "libtard": [1,0,0,0,1,1], "magat": [1,0,0,0,1,1],
    "clown": [1,0,0,0,1,0], "npc": [1,0,0,0,1,0],
    "groomer": [1,1,0,0,1,0], "boomer": [1,0,0,0,1,0],
    "karen": [1,0,0,0,1,0], "mid": [1,0,0,0,1,0],
    "trash": [1,0,0,0,1,0], "garbage": [1,0,0,0,1,0]
}

# Tạo các câu độc hại với slang hiện đại (kèm lỗi chính tả)
for word, lbl in modern_insults.items():
    for _ in range(5): 
        add_toxic(f"You are a {word}", lbl)
        add_toxic(f"Shut up {word}", lbl)
        add_toxic(f"Bro is a {word}", lbl)
        # Thêm biến thể với lỗi chính tả
        add_toxic(f"You are a {simulate_typo(word)}", lbl)

# Các câu slang an toàn (không độc hại)
modern_clean = [
    "This is fire", "This is lit", "Based take", "Common W", 
    "No cap", "Bet", "Drip check", "Goated", "Slay queen"
]
for s in modern_clean:
    for _ in range(5): add_clean(s)

# ==============================================================================
# 3. IDENTITY BIAS CORRECTION
# ==============================================================================
# Mục đích: Sửa lỗi bias - model không nên đánh dấu tất cả câu đề cập đến identity là độc hại
# Chỉ những câu có nội dung kỳ thị mới là độc hại
print("Generating Identity Pairs...")

# Danh sách các nhóm identity (chủng tộc, giới tính, tôn giáo)
identities = ["Black", "White", "Asian", "Gay", "Trans", "Muslim", "Jewish", "Women", "Men"]
# Các vị ngữ trung tính (không độc hại)
neutral_predicates = ["people live here", "culture is interesting", "history is long", "are my friends", "exist"]
# Các vị ngữ độc hại (kỳ thị)
toxic_predicates = ["are criminals", "ruin everything", "are disgusting", "should leave", "are inferior"]

# Tạo cặp dữ liệu: câu trung tính và câu độc hại cho mỗi identity
for ident in identities:
    # Câu trung tính - chỉ đề cập đến identity, không có nội dung kỳ thị
    for pred in neutral_predicates:
        add_clean(f"{ident} {pred}")
        add_clean(f"I am dating a {ident} person")
    
    # Câu độc hại - có nội dung kỳ thị identity
    for pred in toxic_predicates:
        add_toxic(f"{ident} {pred}", [1,0,0,0,1,1])  # toxic, insult, identity_hate
        add_toxic(f"I hate {ident} people", [1,0,0,0,1,1])

# ==============================================================================
# 4. FALSE FRIENDS & AMBIGUOUS WORDS
# ==============================================================================
# Mục đích: Xử lý các từ đa nghĩa - cùng một từ có thể độc hại hoặc an toàn tùy ngữ cảnh
# Ví dụ: "sick" có thể là "bệnh" (độc hại) hoặc "tuyệt vời" (slang, an toàn)
print("Generating False Friends...")

# Dictionary: từ -> (câu độc hại, nhãn, câu an toàn)
ambiguous_map = {
    "sick": ("You are sick", [1,0,0,0,1,0], "That trick was sick"),
    "nasty": ("You are nasty", [1,0,0,0,1,0], "He has a nasty curveball"),
    "filthy": ("Filthy animal", [1,0,0,0,1,0], "Filthy rich"),
    "bad": ("You are bad", [1,0,0,0,1,0], "I want it so bad"),
    "crazy": ("Are you crazy?", [1,0,0,0,1,0], "That was crazy good"),
    "hard": ("You try too hard", [1,0,0,0,1,0], "Go hard or go home"),
    "fat": ("You are fat", [1,0,0,0,1,0], "Big fat check")
}

# Tạo cặp dữ liệu cho mỗi từ đa nghĩa
for word, (tox_sent, lbl, clean_sent) in ambiguous_map.items():
    for _ in range(10):
        add_toxic(tox_sent, lbl)  # Câu độc hại
        add_clean(clean_sent)     # Câu an toàn với cùng từ

# ==============================================================================
# 5. RESTORING & EXPANDING BASE GENERATORS
# ==============================================================================
# Mục đích: Bổ sung các từ lóng và profanity cơ bản, cũng như profanity trong ngữ cảnh tích cực
print("Restoring Base Logic...")

# Dictionary các từ lóng độc hại và nhãn tương ứng
slang_dict = {
    "kys": [1,1,0,1,0,0], "kill urself": [1,1,0,1,0,0], 
    "stfu": [1,0,1,0,1,0], "gtfo": [1,0,1,0,0,0],
    "diaf": [1,1,0,1,0,0], "fml": [0,0,1,0,0,0], 
    "wtf": [0,0,1,0,0,0], "omfg": [0,0,1,0,0,0],
    "sob": [1,0,1,0,1,0], "pos": [1,0,1,0,1,0],
    "retard": [1,0,0,0,1,0], "idiot": [1,0,0,0,1,0],
    "moron": [1,0,0,0,1,0], "loser": [1,0,0,0,1,0],
    "slut": [1,0,1,0,1,0], "whore": [1,0,1,0,1,0],
    "nigger": [1,1,1,0,1,1], "faggot": [1,1,1,0,1,1]
}
# Tạo các câu độc hại với slang
for word, lbl in slang_dict.items():
    for _ in range(5): add_toxic(word, lbl)
    add_toxic(f"You are a {word}", lbl)
    add_toxic(f"Shut up {word}", lbl)

# Profanity trong ngữ cảnh tích cực (an toàn)
# Ví dụ: "That's fucking amazing" - từ "fucking" được dùng như intensifier, không độc hại
intensifiers = ["fucking", "damn", "bloody", "freaking", "really", "hella", "super", "goddamn"]
pos_adjectives = ["amazing", "awesome", "good", "great", "cool", "sick", "dope", "clean", "lit", "brilliant", "excellent", "badass"]
nouns = ["idea", "job", "work", "song", "movie", "game", "art"]

# Tạo các câu an toàn với profanity làm intensifier
for i in intensifiers:
    for a in pos_adjectives:
        add_clean(f"That is {i} {a}")
        add_clean(f"You are {i} {a}")
        # Thêm một số biến thể với lỗi chính tả
        if random.random() > 0.8:
            add_clean(f"That is {simulate_typo(i)} {a}")

# ==============================================================================
# 6. SPECIFIC TEMPLATES (SARCASM, THREAT, HATE, EXCLUSION)
# ==============================================================================
# Mục đích: Tạo các mẫu cho các trường hợp đặc biệt khó phân loại
print("Generating Specific Templates...")

# --- THREATS (Đe dọa) ---
# Các câu đe dọa nhẹ (threat nhưng không severe_toxic)
threat_templates = [
    "Watch your back", "I know where you live", "You better be careful when you leave your house",
    "Keep talking and see what happens to you", "You will regret this",
    "Sleep with one eye open", "We are coming for you",
    "You have no idea what is coming your way", "Your days are numbered",
    "I hope something bad happens to you soon", "It would be a shame if something happened to your family"
]
for t in threat_templates:
    for _ in range(15): add_toxic(t, [1,0,0,1,0,0])  # toxic, threat

# Các câu đe dọa nghiêm trọng (severe_toxic + threat)
severe_threats = [
    "I will hunt you down and kill you", "You are dead meat",
    "Die in a fire", "I'm going to slit your throat",
    "I will find you and end you"
]
for t in severe_threats:
    for _ in range(15): add_toxic(t, [1,1,0,1,0,0])  # toxic, severe_toxic, threat

# Các câu tương tự nhưng an toàn (để model phân biệt được)
clean_similar_threats = [
    "Watch your step, the floor is wet", "Watch your favorite movie tonight",
    "I know where you work, it is a nice building", "I know where the library is",
    "Keep talking, I love listening to your stories", "See what happens in the next episode",
    "Hope something good happens to you soon", "I hope you have a nice day"
]
for c in clean_similar_threats:
    for _ in range(5): add_clean(c)

# --- SARCASM (Mỉa mai) ---
# Các câu mỉa mai, châm biếm - độc hại nhưng không phải threat
sarcastic_templates = [
    "God is so unfair to take away your intelligence",
    "You are proof that evolution can go in reverse",
    "Somewhere a tree is working hard to replace the oxygen you waste",
    "You have the perfect face for radio",
    "I'd agree with you but then we'd both be wrong",
    "You are a waste of space and time",
    "Please try thinking, it's not illegal yet",
    "Of course he failed, what did you expect from someone like that",
    "It is fascinating how you can speak without saying anything smart",
    "Your birth certificate is an apology letter from the condom factory",
    "You are impossible to underestimate"
]
for s in sarcastic_templates:
    for _ in range(15): add_toxic(s, [1,0,0,0,1,0])  # toxic, insult

# --- IMPLICIT HATE (Kỳ thị ngầm) ---
# Các câu kỳ thị không trực tiếp nhưng vẫn có nội dung kỳ thị
hate_templates = [
    "I'm not racist, but they really should stay in their country",
    "They should go back to where they came from",
    "We don't want people like them here",
    "These people are ruining our culture",
    "Why do they act like animals?",
    "I have nothing against them, but they are lazy",
    "Typical behavior from those people",
    "They do not belong in civilized society"
]
for h in hate_templates:
    for _ in range(15): add_toxic(h, [1,0,0,0,0,1])  # toxic, identity_hate

# Các câu tương tự nhưng an toàn
clean_similar_hate = [
    "I'm not a chef, but I love cooking", "They should stay in their country for the holidays",
    "I love visiting their country", "These people are amazing",
    "Typical behavior from a professional athlete", "We want people like them in our team"
]
for c in clean_similar_hate:
    for _ in range(5): add_clean(c)

# --- EXCLUSION (Loại trừ) ---
# Các câu loại trừ, xa lánh người khác
exclusion_templates = [
    "Everyone would be better off if you weren't here",
    "No one wants you around, just leave",
    "Why do you even bother trying?",
    "Nobody likes you", "You are annoying everyone",
    "Please just go away", "Do us a favor and disappear",
    "The world would be a better place without you"
]
for e in exclusion_templates:
    # Một số câu nghiêm trọng hơn (severe_toxic)
    labels = [1,1,0,0,1,0] if "better off" in e or "world" in e else [1,0,0,0,1,0]
    for _ in range(15): add_toxic(e, labels)

# Các câu tương tự nhưng an toàn
clean_exclusion = [
    "Everyone would be better off if we finished early",
    "No one wants to work late", "Just leave the package at the door",
    "Please go away, I need to focus", "Do us a favor and double check the code"
]
for c in clean_exclusion:
    for _ in range(5): add_clean(c)

# ==============================================================================
# 7. SKILL CONTEXT (FIX "YOU ARE KILLER")
# ==============================================================================
# Mục đích: Sửa lỗi model đánh dấu "You are a killer at chess" là độc hại
# Phân biệt: "killer at [skill]" (an toàn) vs "killer" (độc hại)
print("Generating Skill Context...")

skills = ["chess", "coding", "football", "guitar", "gaming", "everything"]
nouns = ["killer", "beast", "monster", "demon", "wizard"]

# Case CLEAN: Khen kỹ năng - "You are a killer at chess" là an toàn
for n in nouns:
    for s in skills:
        add_clean(f"You are a {n} at {s}")
        add_clean(f"He is a {n} at {s}")
add_clean("That was a killer move")
add_clean("Killer app")

# Case TOXIC: Gọi là kẻ giết người (không có 'at') - độc hại
add_toxic("You are a killer", [1,0,0,1,0,0])  # threat
add_toxic("He is a killer", [1,0,0,1,0,0])
add_toxic("They are killers", [1,0,0,1,0,0])
add_toxic("She is a monster", [1,0,0,0,1,0])  # insult

# ==============================================================================
# 8. OBFUSCATION (Che giấu từ ngữ độc hại)
# ==============================================================================
# Mục đích: Xử lý các trường hợp người dùng cố tình che giấu từ ngữ độc hại
# Ví dụ: "f u c k", "f@ck", "f*ck" thay vì "fuck"
print("Generating Obfuscation...")

# Danh sách các từ độc hại cơ bản
toxic_bases = ["fuck", "shit", "bitch", "asshole", "nigger", "faggot", "dick", "cunt"]

def obfuscate(word):
    """
    Tạo các biến thể che giấu của một từ độc hại
    
    Args:
        word (str): Từ gốc cần che giấu
    
    Returns:
        list: Danh sách các biến thể che giấu
            - v1: Thêm khoảng trắng giữa các ký tự ("f u c k")
            - v2: Thay thế ký tự bằng leet speak ("f@ck")
            - v3: Thay 'u' bằng '*'
            - v4: Thay 'i' bằng '*'
    """
    mapping = {'a': '@','i': '1','l': '1','o': '0','s': '$','t': '+','u': 'v','c': '(', 'k': '|<'}
    v1 = " ".join(list(word))  # Thêm khoảng trắng
    v2 = "".join(mapping.get(c, c) for c in word)  # Leet speak
    v3 = word.replace('u', '*')  # Thay ký tự bằng *
    v4 = word.replace('i', '*')
    return [v1, v2, v3, v4]

# Tạo các mẫu độc hại với từ đã được che giấu
for base in toxic_bases:
    variants = obfuscate(base)
    for v in variants:
        for _ in range(5): 
            add_toxic(v, [1,0,1,0,1,0])  # toxic, obscene, insult
            add_toxic(f"You {v}", [1,0,1,0,1,0])

# ==============================================================================
# SAVE (Lưu dữ liệu)
# ==============================================================================
# Chuyển danh sách dữ liệu thành DataFrame
df_aug = pd.DataFrame(data_list)

# Xáo trộn dữ liệu 2 lần để đảm bảo tính ngẫu nhiên
df_aug = df_aug.sample(frac=1, random_state=42).reset_index(drop=True)
df_aug = df_aug.sample(frac=1, random_state=123).reset_index(drop=True)

# Nhân bản x3 để tăng sức nặng của dữ liệu augmented trong training
# Điều này giúp model học tốt hơn các pattern đã được tạo ra
df_final = pd.concat([df_aug, df_aug, df_aug], ignore_index=True)

# Tạo thư mục Data nếu chưa tồn tại
os.makedirs("Data", exist_ok=True)
output_path = "Data/augmented_data.csv"

# Lưu file CSV
df_final.to_csv(output_path, index=False)

print(f"\n✅ DONE! Generated {len(df_final)} samples (Tripled).")
print(f"File saved to: {output_path}")
print(df_final.head(5))