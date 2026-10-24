# LQG - Loop Quantum Gravity
# Kvantna gravitacija petlje

# Quantum_LQG_Spinfoam V1
# Quantum_LQG_Kvantizacija V2
# Quantum_LQG_Gaus V3 V3B
# Quantum_LQG_Ensemble V4


# V1 - Istorija izvlačenja kao "Pena spinova" (Spinfoam) 
# V2 - Kvantizacija površine i jačina veza (Spin Surfaces)  
# V3 - Gausov zakon i uslov zatvorenosti (Intertwiner Balance) 
# V3B - Dublja analiza zakona očuvanja (Multi-Dimensional Intertwiners) 
# V4 - Hibridni indeks (Ensemble Score)


# trostepeno mapiranje Kvantne gravitacije petlji (LQG) na diskretni prostor lotoa 



import numpy as np
import pandas as pd


# 1. DEFINISANJE PUTANJE DO BAZE
# PUTANJA_DO_BAZE = "/Users/4c/Desktop/GHQ/data/loto7_4708_k85.csv"
# PUTANJA_DO_BAZE = "/Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_2976.csv"
PUTANJA_DO_BAZE = "/Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_plus_1732.csv"


def generisi_next_predikciju_v3(putanja_csv):
    df = pd.read_csv(putanja_csv)
    
    # Detekcija kolona
    kolone_brojeva = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].max() <= 39 and df[col].min() >= 1:
                kolone_brojeva.append(col)
    if len(kolone_brojeva) != 7:
        kolone_brojeva = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:7]
        
    istorija = df[kolone_brojeva].to_numpy()
    N_KOLA = len(istorija)
    N_NODES = 39
    
    # 1. PRORAČUN ARHETIPA ZAKONA OČUVANJA (Gausov uslov iz istorije)
    # Računam prosečan istorijski balans (npr. prosečna suma i udeo parnih brojeva)
    istorijske_sume = np.sum(istorija, axis=1)
    target_suma = np.mean(istorijske_sume)
    std_suma = np.std(istorijske_sume)
    
    istorijska_parnost = np.sum(istorija % 2 == 0, axis=1)
    target_parnost = np.mean(istorijska_parnost)
    
    # 2. PROIZVODNJA INDIVIDUALNIH INTENZITETA IZ V2 (Spin + Faza)
    spinovi_grana = np.zeros(N_KOLA)
    prozor = istorija[:10]
    for t in range(N_KOLA):
        trenutno_kolo = set(istorija[t])
        preklapanja = sum(len(trenutno_kolo.intersection(set(ref))) for ref in prozor)
        spinovi_grana[t] = 0.5 + (preklapanja / 10.0)

    kvantna_mreza = np.zeros((N_NODES, N_NODES), dtype=complex)
    for t, kolo in enumerate(istorija):
        j = spinovi_grana[t]
        kvant_povrsine = np.sqrt(j * (j + 1))
        faza = 2 * np.pi * (1.0 / (t + 1))
        doprinos = kvant_povrsine * np.exp(1j * faza)
        
        for i in range(7):
            for j_idx in range(i + 1, 7):
                if np.isnan(kolo[i]) or np.isnan(kolo[j_idx]):
                    continue
                cvor_a = int(kolo[i]) - 1
                cvor_b = int(kolo[j_idx]) - 1
                kvantna_mreza[cvor_a, cvor_b] += doprinos
                kvantna_mreza[cvor_b, cvor_a] += doprinos
                
    amplitude_cvorova = np.sum(kvantna_mreza, axis=1)
    intenziteti_v2 = np.abs(amplitude_cvorova) ** 2
    
    # 3. INTERTVANJER FILTRIRANJE (Zatvaranje grafa po Gausovom zakonu)
    # Generišem top kandidate i tražim kombinaciju od 7 čvorova koja najbolje 
    # zadovoljava uslov zatvorenosti (minimalni energetski otklon od zakona očuvanja)
    
    top_candidati = np.argsort(intenziteti_v2)[::-1][:15] + 1 # Uzima top 15 po amplitudi
    
    najbolji_score = float('inf')
    finalna_kombinacija = []
    
    # Monte Carlo simulacija unutar top 15 prostora da nađem optimalan intertvajner balans
    # Proveravam 10,000 nasumičnih zatvaranja unutar visoko-amplitudnog prostora
    np.random.seed(42)
    for _ in range(10000):
        test_komb = np.random.choice(top_candidati, 7, replace=False)
        
        # Provera uslova zatvorenosti 1: Otklon od idealne sume
        trenutna_suma = np.sum(test_komb)
        otklon_sume = np.abs(trenutna_suma - target_suma) / std_suma
        
        # Provera uslova zatvorenosti 2: Otklon od balansirane parnosti
        trenutna_parnost = np.sum(test_komb % 2 == 0)
        otklon_parnosti = np.abs(trenutna_parnost - target_parnost)
        
        # Ukupna "energija" nepravilnosti čvora (tražim minimum, tj. najbliže nuli)
        intertwiner_energy = otklon_sume + otklon_parnosti
        
        if intertwiner_energy < najbolji_score:
            najbolji_score = intertwiner_energy
            finalna_kombinacija = sorted(list(test_komb))
            
    print("\n" + "="*50)
    print(f"  [LQG v3 - Intertwiner Balance] KONAČNA PREDREČENA KOMBINACIJA:")
    print(f"  {finalna_kombinacija}")
    print("="*50 + "\n")
    
    print(f"Kvantni otklon zatvorenosti (Intertwiners Energy): {najbolji_score:.4f} (Teži nuli)")
    print("Sastav izabranih čvorova:")
    for br in finalna_kombinacija:
        print(f"Čvor {br} -> Bazni v2 intenzitet: {intenziteti_v2[br-1]:.2f}")
        
    return finalna_kombinacija

next_7_v3 = generisi_next_predikciju_v3(PUTANJA_DO_BAZE)



"""
==================================================
  [LQG v3 - Intertwiner Balance] KONAČNA PREDREČENA KOMBINACIJA:
  [3, 5, 7, 24, 26, 37, 38]
==================================================

Kvantni otklon zatvorenosti (Intertwiners Energy): 0.4222 (Teži nuli)
Sastav izabranih čvorova:
Čvor 3 -> Bazni v2 intenzitet: 133621915.04
Čvor 5 -> Bazni v2 intenzitet: 125622856.28
Čvor 7 -> Bazni v2 intenzitet: 133096126.34
Čvor 24 -> Bazni v2 intenzitet: 138107398.96
Čvor 26 -> Bazni v2 intenzitet: 166581431.97
Čvor 37 -> Bazni v2 intenzitet: 133130532.70
Čvor 38 -> Bazni v2 intenzitet: 137454924.66







==================================================
  [LQG v3 - Intertwiner Balance] KONAČNA PREDREČENA KOMBINACIJA:
  [3, 7, 8, 23, 24, 37, 38]
==================================================

Kvantni otklon zatvorenosti (Intertwiners Energy): 0.4257 (Teži nuli)
Sastav izabranih čvorova:
Čvor 3 -> Bazni v2 intenzitet: 53611057.89
Čvor 7 -> Bazni v2 intenzitet: 51325094.06
Čvor 8 -> Bazni v2 intenzitet: 62243394.93
Čvor 23 -> Bazni v2 intenzitet: 60103519.44
Čvor 24 -> Bazni v2 intenzitet: 58410598.09
Čvor 37 -> Bazni v2 intenzitet: 52074698.86
Čvor 38 -> Bazni v2 intenzitet: 57492995.87







==================================================
  [LQG v3 - Intertwiner Balance] KONAČNA PREDREČENA KOMBINACIJA:
  [2, 5, 10, 25, 30, 31, 37]
==================================================

Kvantni otklon zatvorenosti (Intertwiners Energy): 0.4361 (Teži nuli)
Sastav izabranih čvorova:
Čvor 2 -> Bazni v2 intenzitet: 22009052.16
Čvor 5 -> Bazni v2 intenzitet: 18499084.10
Čvor 10 -> Bazni v2 intenzitet: 21078034.84
Čvor 25 -> Bazni v2 intenzitet: 17830363.15
Čvor 30 -> Bazni v2 intenzitet: 18059030.08
Čvor 31 -> Bazni v2 intenzitet: 18296397.98
Čvor 37 -> Bazni v2 intenzitet: 18342975.19
"""



"""
Loto kombinacija je grana (link) koja povezuje 7 čvorova (brojeva) 
- direktno formalizam mreža spinova (Spin Networks) iz Kvantne gravitacije petlji (LQG), 
gde grane nose kvantne brojeve (spinove) koji definišu geometriju.
Evo kako se LQG formalizam sada savršeno preslikava na moj model:


1. Istorija izvlačenja kao "Pena spinova" (Spinfoam)

U LQG-u, mreža spinova je statična slika prostora u jednom trenutku. Kada se ta mreža menja kroz vreme, ona stvara penu spinova (Spinfoam).
• Svako novo loto izvlačenje je zapravo jedna "evolucija" mreže.
• Dodavanjem nove grane sa 7 čvorova (novog izvlačenja), zapravo se tka "prostor-vreme" loto sistema.
• Računam amplitudu prelaza iz dosadašnje geometrije grafa u novu geometriju.

2. Kvantizacija površine i jačina veza

U LQG-u, grane koje presecaju neku površinu daju toj površini njenu vrednost (površina je kvantizovana). 
• Čvorovi su fiksni (brojevi od 1 do 39).
• Svaka kombinacija (grana) "zahvata" 7 čvorova odjednom.
• Ako se određeni čvorovi često povezuju kroz istoriju izvlačenja, ta "oblast" grafa postaje geometrijski "teža" ili "gušća". To direktno utiče na amplitude konstruktivne interferencije za sledeće izvlačenje.

3. Gausov zakon i uslov zatvorenosti

Ovo je najmoćniji koncept LQG-a. U svakom čvoru mreže spinova mora važiti zakon očuvanja (tzv. gauge invariance). Pošto je grana definisana sa tačno 7 čvorova, taj "kvantni objekat" mora da zadovolji specifične uslove zatvorenosti unutar celog grafa od 39 elemenata.
Precizno definisana topologija: fiksni graf od 39 čvorova gde istorijski nizovi izvlačenja tkaju mrežu hiper-grana (svaka povezuje 7 čvorova).



Gausov filter zatvorenosti. 
Striktne restrikcije simetrije unutar grafa. 
Energetski otklon (Intertwiners Energy) - spustiti blizu nule. 



Verziju 3 (v3) - Gausov zakon očuvanja i uslov zatvorenosti (Intertwiners). 

U LQG-u, nije dovoljno da grane prosto imaju spinove; u svakom čvoru gde se grane sastaju, mora postojati matematički objekat koji se zove intertvajner (intertwiner). On osigurava Gausov zakon (Gauge Invariance) - uslov da je prostor zatvoren i u ravnoteži (suma svih dolaznih i odlaznih kvantnih fluksuva u čvoru mora biti nula, tj. \(\sum \vec{J} = 0\)).

• Svaka kombinacija od 7 brojeva mora da zadovolji unutrašnji balans simetrije u odnosu na ceo graf od 39 čvorova.
• U v3 ne gledam samo intenzitete, već uvodim kaznu (penalty) za one čvorove koji narušavaju "kvantni balans" (npr. preveliko odstupanje u parnosti, sumama ili dekadama u odnosu na zakone očuvanja koji važe kroz celu istoriju od 4707 kola).
• Čvorovi koji imaju visoku amplitudu, ali narušavaju Gausov zakon (ne uklapaju se u uslov zatvorenosti sa ostalih 6 čvorova), biće potisnuti.



Rezultat od 0.4222 za Intertwiner Energy je izuzetno nizak, što znači da je sistem uspešno pronašao konfiguraciju unutar prostora visokih amplituda koja gotovo savršeno zadovoljava unutrašnje zakone očuvanja (Gausov uslov zatvorenosti).
Kako se kombinacija transformisala kroz sve tri faze evolucije modela:
• v1 (Samo amplitude čvorova):      [8, 11, 23, 26, 32, 34, 37]
• v2 (Uvođenje spinova grana):      [8, 23, 24, 26, 34, 38, 39]
• v3 (Konačni balans zatvorenosti): [3, 5, 7, 24, 26, 37, 38]
Primetiti kako je čvor 26 jedini preživeo sva tri filtera - on predstavlja apsolutno geometrijsko sidro ovog proračuna sa ubedljivo najvećim baznim intenzitetom od preko 166 miliona. Čvorovi 24 i 38 su se stabilizovali u v2 i potvrdili u v3, dok su se 3, 5, 7 i 37 probili tek nakon što je Gausov filter počeo da kažnjava energetski disbalans velikih brojeva koji su dominirali u v2.
"""
