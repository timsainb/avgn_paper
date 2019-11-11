#https://buckeyecorpus.osu.edu/SpeechSearcherManual.pdf
"""
C - All consonants. See section 3.7 for details.
• P - Plosives (=stops): p, t, k, b, d, g, tq
• J - Affricates: ch, jh
• N - Nasals: m, n, ng
• F - Fricatives: f, v, th, dh, s, z, sh, zh, hh
• L - Liquids: l, el, r, er, ern
• Y - Glides (=semivowels): w, y
• LAB - Labials: p, b, m, f, v
• COR - Coronals: th, dh, t, d, n, s, z, sh, zh, r, er, ern, l, el
• DOR - Dorsals: y, k, g, ng
• LAR - Laryngeals: hh


• V - All vowels. See section 3.7 for details.
• DP - Diphthongs: (± nasal) ay, ey, oy, aw, ow
• LV - Long vowels: (± nasal) iy, aa, ao, uw
• SV - Short vowels: (± nasal) ih, eh, ae, ah, uh
• HV - High vowels: (± nasal) iy, ih, uh, uw
• MV - Mid vowels: (± nasal) eh, ah, ao
• LWV - Low vowels: (± nasal) ae, aa
• FV - Front vowels: (± nasal) iy, ih, ey, eh, ae
• CENV - Central vowels: (± nasal) ah
• BV - Back vowels: (± nasal) uw, uh, ow, ah, aa
• ER - Syllabic consonants: (± nasal) em, en, el, er
"""

VOWEL_CONSONANT = {
    'consonants': {
        'Plosives': ['p', 't', 'k', 'b', 'd', 'g', 'tq'],
        'Affricates': ['ch', 'jh'],
        'Nasals': ['m', 'n', 'ng'],
        'Fricatives': ['f', 'v', 'th', 'dh', 's', 'z', 'sh', 'zh', 'hh'],
        'Liquids': ['l', 'el', 'r', 'er', 'ern'],
        'Glides': ['w', 'y'],
        'Labials': ['p', 'b', 'm', 'f', 'v'],
        'Coronals': ['th', 'dh', 't', 'd', 'n', 's', 'z', 'sh', 'zh', 'r', 'er', 'ern', 'l', 'el'],
        'Dorsals': ['y', 'k', 'g', 'ng'],
        'Laryngeals': ['hh']
    },
    'vowels': {
        'Diphthong': ['ay', 'ey', 'oy', 'aw', 'ow'],
        'Long vowels': ['iy', 'aa', 'ao', 'uw'],
        'Short vowels': ['ih', 'eh', 'ae', 'ah', 'uh'],
        'High vowels': ['iy', 'ih', 'uh', 'uw'],
        'Mid vowels': ['eh', 'ah', 'ao'],
        'Low vowels': ['ae', 'aa'],
        'Front vowels': ['ih', 'ey', 'eh', 'ae'],
        'Central vowels': ['ah'],
        'Back vowels': ['uw', 'uh', 'ow', 'ah', 'aa'],
        'Syllabic consonants': ['em', 'en', 'el', 'er']
    }
    
}