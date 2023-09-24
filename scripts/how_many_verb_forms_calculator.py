def count_latin_verb_forms():
    persons = 3  # 1st, 2nd, 3rd
    numbers = 2  # singular, plural
    voices = 2   # active, passive
    indicative_tenses = 6  # present, imperfect, future, perfect, pluperfect, future perfect
    subjunctive_tenses = 4  # present, imperfect, perfect, pluperfect
    imperative_forms = 6  # 2nd person singular and plural, active and passive; 3rd person active and passive

    total_forms = (persons * numbers * voices * indicative_tenses) + (persons * numbers * voices * subjunctive_tenses) + imperative_forms

    return total_forms

print(f"Total regular Latin verb forms: {count_latin_verb_forms()}")

