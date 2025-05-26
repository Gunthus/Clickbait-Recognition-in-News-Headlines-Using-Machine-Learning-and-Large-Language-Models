"""
Custom Latvian stopword list (based on NLL resources).
Note: we intentionally omit 'kas' and 'kā' so they remain as potential clickbait cues.
As mentioned in the thesis, these words are kept because they correlate with clickbait structure.
"""

STOPWORDS = {
    'un', 'vai', 'bet', 'ar', 'bez', 'līdz', 'pēc', 'no', 'uz', 'par',
    'jo', 'kad', 'kur', 'ko', 'lai', 'tad', 'tikai', 'jau', 'tas',
    'šis', 'šī', 'viņš', 'viņa', 'es', 'tu', 'mēs', 'jūs', 'viņi',
    'mans', 'tavs', 'mūsu', 'jūsu', 'man', 'tev', 'mums', 'jums',
    'ir', 'bija', 'būs', 'esmu', 'esi', 'esat', 'esam',
    'caur', 'starp', 'pati', 'pašlaik', 'tostarp', 'vien',
    'vis', 'viens', 'nekā', 'nekad', 'vienīgi', 'gan', 'tik', 'tur',
    'turklāt', 'pret', 'kopā', 'vidū', 'zem', 'virs'
}