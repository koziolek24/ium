import matplotlib.pyplot as plt

# Dane z dokumentacji
data = {
    'neighbourhood': {
        'NaN': 17858, 'Bangkok, Krung Thep...': 755, 'Khet Khlong Toei...': 746,
        'Bangkok, Thailand': 577, 'Khet Watthana...': 532, 'Watthana...': 445
    },
    'neighbourhood_cleansed': {
        'NaN': 5137, 'Vadhana': 3278, 'Khlong Toei': 2841, 
        'Huai Khwang': 2118, 'Ratchathewi': 1257, 'Sathon': 976
    },
    'property_type': {
        'Entire rental unit': 8118, 'NaN': 5055, 'Entire condo': 3412,
        'Private room...': 2174, 'Room in hotel': 1392, 'Entire home': 915
    },
    'room_type': {
        'Entire home/apt': 13475, 'Private room': 6608, 'NaN': 5041,
        'Hotel room': 341, 'Shared room': 144
    },
    'accommodates': {
        '2.0': 11921, 'NaN': 5087, '4.0': 3044, 
        '3.0': 2280, '6.0': 917, '5.0': 578
    },
    'bathrooms': {
        '1.0': 11797, 'NaN': 9206, '2.0': 2107, 
        '1.5': 852, '3.0': 474, '4.0': 253
    },
    'bedrooms': {
        '1.0': 13893, 'NaN': 6099, '2.0': 3251, 
        '3.0': 857, '0.0': 718, '4.0': 395
    },
    'beds': {
        '1.0': 9710, 'NaN': 9201, '2.0': 3658, 
        '3.0': 1112, '4.0': 639, '0.0': 405
    }
}

# Generowanie i zapisywanie każdego wykresu osobno
for attr, values in data.items():
    plt.figure(figsize=(10, 6))
    labels = list(values.keys())
    counts = list(values.values())
    
    bars = plt.bar(labels, counts, color='teal', alpha=0.7)
    plt.title(f'Rozkład atrybutu: {attr}', fontsize=14, fontweight='bold')
    plt.ylabel('Liczba wystąpień')
    plt.xticks(rotation=30, ha='right')
    
    # Dodanie wartości nad słupkami
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    # Zapis do pliku
    plt.savefig(f'wykres_{attr}.png')
    plt.close() # Zamknięcie figury, aby zwolnić pamięć