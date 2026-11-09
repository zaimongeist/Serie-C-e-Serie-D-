"""Glossary ("Legenda") of every quantity the dashboard, the console and the CSV exports show.

The content is data (:data:`LEGEND`, a tuple of :class:`LegendSection`), written
in Italian for someone who uses the numbers to plan bets, and rendered to
Markdown by :func:`legend_markdown` -- the same text feeds the *Legenda* tab of
the Streamlit app and the *Legenda* section of ``README.md`` (a test keeps the
two in sync). :func:`worked_example_markdown` explains one row of the value
panel with real numbers (:class:`WorkedExample`, built from a
:class:`~src.engine.value.ValueOpportunity` or synthetic).

No ``streamlit`` import: the module is pure so it can be unit-tested and reused
by ``main.py``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from src.engine.value import (
    FLAG_LONGSHOT,
    FLAG_REGRESSION_TRAP,
    FLAG_SUSPICIOUS_EDGE,
    FLAG_VALUE,
    KELLY_EIGHTH,
    TRAP_FLAG_PREFIX,
    KellyConfig,
    ValueOpportunity,
    kelly_fraction,
    stake_units,
)

__all__ = [
    "LEGEND",
    "LEGEND_TITLE",
    "POOR_EXAMPLE_FLAGS",
    "SYNTHETIC_REASON_NO_QUOTES",
    "SYNTHETIC_REASON_NO_VALUE",
    "LegendEntry",
    "LegendSection",
    "WorkedExample",
    "legend_markdown",
    "legend_sections",
    "legend_terms",
    "pick_example_row",
    "worked_example_markdown",
]

#: Heading of the whole legend (``## `` level in Markdown, so it nests under the README's sections).
LEGEND_TITLE: Final[str] = "Legenda: come leggere i numeri"
#: Reasons a :class:`WorkedExample` is synthetic, printed in its header (see :meth:`WorkedExample.synthetic`).
SYNTHETIC_REASON_NO_QUOTES: Final[str] = "nessuna quota nei dati correnti"
SYNTHETIC_REASON_NO_VALUE: Final[str] = "nessuna selezione di valore con i parametri correnti"


@dataclass(frozen=True)
class LegendEntry:
    """One glossary line: the ``term`` as it appears on screen and its ``meaning`` in Italian."""

    term: str
    meaning: str

    def __post_init__(self) -> None:
        if not self.term.strip() or not self.meaning.strip():
            raise ValueError("a legend entry needs both a term and a meaning")

    def to_markdown(self) -> str:
        return f"- **{self.term}** — {self.meaning}"


@dataclass(frozen=True)
class LegendSection:
    """A titled group of :class:`LegendEntry` (``bullets=True`` renders terms as plain bullets, for rules)."""

    title: str
    entries: tuple[LegendEntry, ...]
    intro: str = ""

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("a legend section needs a title")
        if not self.entries:
            raise ValueError(f"legend section {self.title!r} has no entries")

    def to_markdown(self) -> str:
        lines = [f"### {self.title}"]
        if self.intro:
            lines += ["", self.intro]
        lines.append("")
        lines += [entry.to_markdown() for entry in self.entries]
        return "\n".join(lines)


def _e(term: str, meaning: str) -> LegendEntry:
    return LegendEntry(term, meaning)


LEGEND: Final[tuple[LegendSection, ...]] = (
    LegendSection(
        "Parametri del modello (tab Classifica Forza)",
        (
            _e(
                "α (alfa) · Forza offensiva",
                "quanto una squadra segna più o meno della media della sua competizione, su scala logaritmica: 0 = attacco "
                "medio del girone, +0.3 ≈ 35% di gol in più (exp(0.3) = 1.35), −0.3 ≈ 26% in meno. È centrata per "
                "competizione (la somma degli α di un girone è zero): confronta gli α solo dentro la stessa competizione, "
                "un +0.2 in Serie B e un +0.2 nel girone C non misurano la stessa cosa.",
            ),
            _e(
                "β (beta) · Solidità difensiva",
                "quanto una squadra fa segnare poco gli avversari: più alto = subisce meno. Entra nei gol attesi "
                "dell'avversario con il segno meno (λ = exp(α_casa − β_ospite + γ)). A differenza di α, β NON è centrata "
                "su zero: la media dei β di una competizione fa da livello base dei gol di quel campionato, quindi il "
                "valore assoluto non dice nulla e β = 0 non significa \"difesa media\". Leggila in confronto: 0.3 di β in "
                "più di un'altra squadra della stessa competizione ≈ 26% di gol subiti in meno (exp(−0.3) = 0.74); sopra "
                "la media dei β del girone = difesa migliore della media, sotto = peggiore.",
            ),
            _e(
                "γ (gamma) · Vantaggio casa",
                "fattore campo stimato per ogni competizione (uno per girone, uno per la Serie B), non per squadra: i gol "
                "attesi della squadra di casa sono moltiplicati per exp(γ). γ = 0.25 ⇒ ×1.28, cioè circa il 28% di gol in "
                "più di quanti ne farebbe in campo neutro. Valori tipici 0.15-0.35.",
            ),
            _e(
                "Indice Attacco / Indice Difesa",
                "exp(α) ed exp(β): gli stessi parametri in scala moltiplicativa. Indice Attacco 1.00 = attacco medio del "
                "girone, 1.20 = +20%, 0.85 = −15%. L'Indice Difesa non ha 1.00 come media (β non è centrata): si legge "
                "per rapporto, se una squadra ha indice difesa 1.20 volte quello di un'altra della stessa competizione, "
                "a parità di avversario subisce circa 1/1.20 ≈ 17% di gol in meno.",
            ),
            _e(
                "ρ (rho) · Correlazione bassi punteggi",
                "correzione di Dixon-Coles ai risultati 0-0, 1-0, 0-1 e 1-1. Con ρ negativo (tipico, tra −0.05 e −0.15) "
                "il modello prevede più 0-0 e 1-1 (e meno 1-0 / 0-1) di due Poisson indipendenti: i pareggi a basso "
                "punteggio salgono, con ρ = −0.10 la X guadagna circa 2-3 punti percentuali. ρ = 0 equivale a gol di casa "
                "e ospiti indipendenti. È unico per tutto il perimetro.",
            ),
            _e(
                "ξ (xi) · Decadimento temporale",
                "peso delle partite nel tempo: ogni gara pesa exp(−ξ·giorni trascorsi). Con ξ = 0.0018 una partita di un "
                "anno fa vale circa il 52% di una di oggi, una di due anni fa il 27%: lo storico delle tre stagioni aiuta "
                "la stima ma conta sempre meno della stagione in corso. ξ = 0 (dati simulati) pesa tutto allo stesso modo.",
            ),
            _e(
                "Prior gaussiana · Stima MAP",
                "a inizio stagione ci sono poche partite e la stima «pura» (massima verosimiglianza, MLE) esagera: una "
                "squadra con due vittorie 4-0 sembrerebbe imbattibile. La prior tira i parametri verso valori normali "
                "(α verso 0 e β verso la media del suo girone, entrambi con deviazione standard 0.30, γ verso 0.25 ± 0.15, ρ verso "
                "0 ± 0.10) e la stima diventa "
                "«MAP» (massimo a posteriori). Si attiva da sola finché la competizione meno coperta ha meno di 15 partite "
                "giocate per squadra nella stagione corrente (con 20 squadre per girone: fino alla 14ª giornata inclusa); "
                "il badge della sidebar dice se è attiva. Con una stagione intera i dati dominano e MAP ≈ MLE.",
            ),
        ),
    ),
    LegendSection(
        "Previsione della partita (tab Giornata)",
        (
            _e(
                "λ (lambda) · Gol attesi casa",
                "numero medio di gol che il modello si aspetta dalla squadra di casa: λ = exp(α_casa − β_ospite + γ). "
                "Un λ di 1.60 non vuol dire «segna 1 o 2 gol», ma che in media su molte partite così ne segnerebbe 1.6: "
                "la distribuzione dei singoli punteggi è nella matrice.",
            ),
            _e(
                "μ (mu) · Gol attesi ospiti",
                "lo stesso per la squadra in trasferta: μ = exp(α_ospite − β_casa), senza fattore campo. λ + μ è il "
                "totale gol atteso della partita (sopra 2.8 = gara da Over, sotto 2.2 = gara chiusa).",
            ),
            _e(
                "Matrice dei punteggi (heatmap)",
                "probabilità di ogni risultato esatto, gol casa sulle righe e gol ospiti sulle colonne, da 0 a 10 gol per "
                "squadra (la vista compatta accorpa la coda in «N+»). Le celle sommano al 100%: ogni mercato è solo una "
                "somma di celle (1 = celle sotto la diagonale, X = diagonale, Over 2.5 = celle con somma ≥ 3, GG = celle "
                "con riga ≥ 1 e colonna ≥ 1).",
            ),
            _e(
                "Risultato più probabile",
                "la cella più alta della matrice, con la sua probabilità. Di norma vale il 10-14%: è il punteggio singolo "
                "più frequente, non «il risultato previsto». Un 1-1 al 12% convive con un 1 (vittoria casa) al 45%.",
            ),
            _e(
                "Probabilità dei mercati · 1X2, Over/Under, GG/NG",
                "per ogni mercato gli esiti sono esaustivi e si escludono a vicenda, quindi sommano al 100% (1 + X + 2, "
                "Over 2.5 + Under 2.5, GG + NG). Over 1.5 / 2.5 / 3.5 contano i gol totali (strettamente più di 1, 2, 3); "
                "GG = «entrambe le squadre segnano», NG = almeno una resta a zero.",
            ),
        ),
    ),
    LegendSection(
        "Tabella giornata e risultati",
        (
            _e(
                "Tabella giornata (tab Giornata)",
                "una riga per **ogni partita** della giornata ufficiale di ogni competizione (tutte le competizioni della "
                "sezione o quella della sidebar): le partite già giocate con il risultato finale, quelle da giocare con orario e "
                "previsione del modello (forza α/β, forma recente, gol attesi, risultato più probabile, probabilità e quote fair "
                "dei mercati, miglior quota book per esito, EV migliore e flag), quelle già iniziate senza risultato nei dati e "
                "quelle rinviate. Tutto il turno si legge in un "
                "colpo d'occhio, anche quando una parte è già stata giocata il venerdì; cliccando una partita da giocare (o "
                "scegliendola nel menù «Partita») sotto compaiono la matrice dei punteggi e i mercati di quella partita. "
                "«Scarica tabella giornata (CSV)» ha le stesse colonne più match_id.",
            ),
            _e(
                "Giornata corrente",
                "la giornata ufficiale (game_week di FootyStats) che si sta giocando o che sta per iniziare, competizione per "
                "competizione: la più bassa che ha ancora almeno una partita in programma (calcio d'inizio non più vecchio di 3 "
                "ore: una partita iniziata da meno di 3 ore tiene la giornata corrente) e in cui almeno metà delle partite con "
                "una data non è ancora giocata (le rinviate senza nuova data non contano), oppure che è in corso: qualche "
                "partita giocata o iniziata e una partita rimasta nello stesso blocco della settimana (venerdì-lunedì o "
                "martedì-giovedì) della maggior parte del turno. Così un turno con due partite giocate il venerdì e otto nel "
                "fine settimana resta la giornata corrente, con tutte e dieci le partite; un turno di cui resta solo un recupero "
                "infrasettimanale no (anche se altre partite sono rinviate): la giornata corrente è la successiva e il recupero "
                "va in «Recuperi e rinvii».",
            ),
            _e(
                "Calcio d'inizio",
                "giorno e ora della partita in ora italiana, «sab 07/11 14:30» (FootyStats dà l'orario in UTC: il passaggio fra "
                "ora legale e solare è già considerato). Le partite giocate e i file senza orario mostrano solo il giorno, "
                "«ven 06/11».",
            ),
            _e(
                "Stato · giocata / da giocare / in corso / rinviata",
                "«giocata 2-1» = partita già disputata, con il risultato finale e le colonne del modello vuote (non si prevede "
                "il passato); «da giocare» = calcio d'inizio ancora da venire, con la previsione: sono le sole partite delle "
                "selezioni e del pannello valore; «in corso / risultato in attesa» = il calcio d'inizio è passato ma il "
                "risultato non è ancora nei dati (partita in corso, FootyStats non l'ha ancora pubblicato o i dati locali sono "
                "di prima della fine): resta nel suo turno, senza previsione e senza quote, perché una giocata pre-partita su "
                "una partita iniziata non ha senso; «rinviata» = nessun risultato 48 ore dopo il calcio d'inizio e recupero non "
                "ancora fissato: niente previsione finché non ha una nuova data.",
            ),
            _e(
                "Giornata n. (una competizione)",
                "con una sola competizione scelta nella sidebar compare il menù «Giornata n.»: mostra qualsiasi giornata "
                "ufficiale di quella competizione, passata o futura, con tutte le sue partite; «(corrente)» indica la giornata "
                "corrente.",
            ),
            _e(
                "Recuperi e rinvii",
                "l'elenco, sotto la tabella, delle partite rinviate e dei recuperi ancora da giocare fissati fuori dalla giornata "
                "corrente della loro competizione (per esempio un recupero del mercoledì di un turno già giocato), con la nuova "
                "data e, per una partita spostata davvero, la data originale (il giorno in cui si giocava il resto del turno). "
                "Non spostano mai la giornata e non entrano nelle selezioni di giornata; per quelle in programma sono indicate le "
                "probabilità 1X2 e GG del modello. Una partita già iniziata senza risultato non è un recupero: resta nel suo "
                "turno come «in corso / risultato in attesa».",
            ),
            _e(
                "Forma (Casa / Ospiti)",
                "gli esiti delle ultime 5 partite giocate dalla squadra nella stagione corrente, casa e trasferta insieme, "
                "dalla più recente alla più vecchia: V vittoria, N pareggio (nullo), P sconfitta. «V-V-N-P-N» = ha vinto le "
                "ultime due. Meno di 5 lettere = ha giocato meno di 5 partite; vuota = nessuna partita giocata.",
            ),
            _e(
                "Pt (5)",
                "punti raccolti nelle stesse ultime 5 partite (3 per vittoria, 1 per pareggio): da 0 a 15; il numero fra "
                "parentesi nel nome della colonna è la finestra. Non è la classifica: misura il momento di forma, che il "
                "modello non usa direttamente (α e β pesano tutta la stagione, più lo storico).",
            ),
            _e(
                "GF/GS (5)",
                "gol fatti / gol subiti nelle ultime 5 partite: «9/4» = 9 segnati e 4 subiti. Confrontalo con λ e μ: una "
                "squadra che segna molto più di quanto il modello si aspetti sta sovra-performando (o il modello la "
                "sottovaluta), una che subisce poco con tanti xG contro è candidata alla trappola di regressione.",
            ),
            _e(
                "xG F/S (3)",
                "expected goals prodotti / concessi nelle ultime 3 partite: «4.1/2.6» = 4.1 xG creati e 2.6 subiti; «-» "
                "quando il dato manca anche in una sola delle tre gare. La finestra di 3 è la stessa del filtro "
                "REGRESSION_TRAP: xG subiti alti con pochi gol subiti (GF/GS) = difesa in sovra-rendimento.",
            ),
            _e(
                "Tot. att.",
                "λ + μ, i gol totali attesi della partita secondo il modello: sopra 2.8 gara da Over, sotto 2.2 gara "
                "chiusa. È la stessa somma che guida le probabilità Over/Under e GG delle colonne accanto.",
            ),
            _e(
                "Ris. prob.",
                "il risultato esatto più probabile con la sua probabilità, «1-0 (14%)»: la cella più alta della matrice "
                "dei punteggi, di norma fra il 10 e il 14%. Non è «il pronostico»: un 1-1 al 13% convive con una vittoria "
                "casa al 45%.",
            ),
            _e(
                "P1 / PX / P2, O1.5 / O2.5 / O3.5, GG (%) · Fair · Book",
                "le probabilità del modello per vittoria casa, pareggio, vittoria ospite, Over 1.5 / 2.5 / 3.5 gol totali ed "
                "entrambe a segno; Fair 1 / X / 2 / O2.5 / GG le quote fair corrispondenti (1/P); Book 1 / X / 2 / O2.5 / GG "
                "la miglior quota fra i bookmaker presenti nei dati per quell'esito (vuota se nessuno lo quota).",
            ),
            _e(
                "EV max (%) · Miglior EV",
                "il valore atteso più alto fra tutte le selezioni quotate della partita (ogni mercato, ogni bookmaker) e la "
                "selezione che lo realizza: «X @3.40» = il pareggio alla quota 3.40. È il punto da cui partire per quella "
                "partita; il dettaglio (bookmaker, edge, stake) è nel pannello Opportunità di Valore. Vuoti senza quote.",
            ),
            _e(
                "Flag (tabella giornata)",
                "l'unione dei flag di tutte le selezioni quotate della partita (VALUE, LONGSHOT, SUSPICIOUS_EDGE: vedi la "
                "sezione Flag) più REGRESSION_TRAP e TRAP:<squadra> quando una delle due squadre è in trappola di "
                "regressione, anche senza quote: il segnale sulla difesa vale a prescindere dal prezzo.",
            ),
            _e(
                "Tabella risultati (tab Risultati)",
                "una riga per ogni partita già giocata della stagione e della competizione scelte, filtrabile per squadra "
                "(il menù offre chi ha giocato quella stagione in quella competizione: nelle stagioni passate anche le squadre "
                "non più nel perimetro, mai una squadra senza partite): risultato, esito 1X2, tiri, tiri in porta "
                "e angoli casa/ospiti («12/7», «-» se il dato manca), xG di entrambe, più la lettura del modello (λ, μ, P "
                "esito, Sorpresa). Le colonne si ordinano cliccando l'intestazione; «Scarica risultati (CSV)» la esporta.",
            ),
            _e(
                "P esito (%)",
                "la probabilità che il modello assegna all'esito 1X2 effettivamente avvenuto, calcolata con λ e μ di quella "
                "partita (la stessa matrice dei punteggi). Attenzione: il modello è stimato su tutte le partite giocate, "
                "compresa quella stessa, quindi è una lettura a posteriori, non una previsione fatta prima del calcio "
                "d'inizio; serve a vedere quali risultati il modello spiega e quali no.",
            ),
            _e(
                "Sorpresa",
                "«sorpresa» quando P esito è sotto il 20% (un esito che il modello riteneva improbabile), «atteso» sopra il "
                "50% (l'esito che il modello indicava come favorito), vuoto nel mezzo. Molte sorprese di fila per una "
                "squadra = qualcosa che il modello non vede (infortuni, cambio allenatore, calendario) oppure semplice "
                "varianza: è un invito a guardare, non un segnale di scommessa.",
            ),
        ),
    ),
    LegendSection(
        "Quote e valore (tab Opportunità di Valore)",
        (
            _e(
                "Quota fair",
                "1 / probabilità del modello: la quota a cui la scommessa sarebbe alla pari nel lungo periodo. P = 40% "
                "⇒ fair 2.50. Se il bookmaker paga più della fair la selezione ha valore, se paga meno no.",
            ),
            _e(
                "Quota book",
                "la quota decimale del bookmaker (colonna Bookmaker = da dove viene: FootyStats, SimBook del simulatore o "
                "il valore della colonna bookmaker del CSV in data/odds). Con più book sulla stessa selezione si usa la "
                "quota più alta.",
            ),
            _e(
                "Probabilità implicita e de-vig",
                "1 / quota è quello che il book «dice» di credere, ma le implicite di un mercato sommano a più del 100% "
                "(es. 105%): il 5% in eccesso è l'allibramento (vig, margine). Il de-vig lo rimuove riproporzionando le "
                "implicite in modo che sommino al 100% (metodo proporzionale di default; power e Shin disponibili). Serve "
                "a confrontare il modello con l'opinione del book depurata dal margine.",
            ),
            _e(
                "EV (%) · Valore atteso",
                "EV = P_modello × Quota book − 1. Esempio: 40% × 2.70 − 1 = +8%: per ogni euro puntato ci si aspetta di "
                "guadagnare in media 8 centesimi **nel lungo periodo**, su molte scommesse simili; sulla singola partita "
                "si vince o si perde. EV negativo = scommessa in perdita attesa anche se vince stavolta.",
            ),
            _e(
                "Edge",
                "P_modello − P_implicita de-viggata: di quanti punti percentuali il modello è più ottimista del "
                "bookmaker su quell'esito. Misura il disaccordo; l'EV misura quanto rende. Un edge grande con quota bassa "
                "vale poco; un edge piccolo su quota alta può valere molto.",
            ),
            _e(
                "Soglia EV (3,5%)",
                "sotto questa soglia la selezione non è considerata di valore: l'EV calcolato ha un errore (quello del "
                "modello) e un margine di sicurezza evita di puntare su differenze che sono solo rumore. Il cursore nella "
                "sidebar la cambia; alzarla riduce le puntate ma le rende più selettive.",
            ),
        ),
    ),
    LegendSection(
        "Dimensionamento della puntata (colonna Stake)",
        (
            _e(
                "Kelly pieno · f* = (p·b − q) / b",
                "frazione del bankroll che massimizza la crescita a lungo termine, con p = P_modello, q = 1 − p e b = "
                "quota − 1 (vincita netta per 1 euro). Esempio: p = 40%, quota 2.70 ⇒ b = 1.70, f* = (0.40×1.70 − 0.60) "
                "/ 1.70 = 4.7% del bankroll. Se p·b < q (EV negativo) f* è zero: non si punta.",
            ),
            _e(
                "Kelly frazionario (1/8 o 1/4)",
                "si punta solo una frazione di f*, perché p è una stima: il Kelly pieno con una probabilità sbagliata "
                "porta a scommesse troppo grosse e a oscillazioni violente del bankroll. 1/8 (default) è prudente e adatto "
                "a un modello nuovo; 1/4 è già aggressivo. La frazione dimezza (o riduce) la varianza molto più di quanto "
                "riduca la crescita attesa.",
            ),
            _e(
                "Stake (unità)",
                "la puntata consigliata in unità di bankroll: con bankroll 100 unità, 1 unità = 1% del capitale dedicato "
                "alle scommesse. Stake = f* × frazione di Kelly × bankroll, arrotondato a 0,1 unità. Nell'esempio sopra "
                "con 1/8 Kelly e 100 unità: 4.7% × 1/8 × 100 = 0,6 unità.",
            ),
            _e(
                "Limite per puntata",
                "lo stake è comunque tappato a 3 unità (il 3% con il bankroll standard di 100 unità; il tappo resta 3 unità "
                "anche cambiando il bankroll nella sidebar): oltre, l'eventuale errore del modello costa troppo. Uno stake "
                "«3.0» dice che il Kelly avrebbe chiesto di più.",
            ),
            _e(
                "Stake 0 unità",
                "la selezione o ha EV sotto soglia (nessun valore) o la puntata arrotondata scende sotto 0,1 unità: non "
                "vale la pena piazzarla. Il pannello valore mostra solo righe sopra soglia, ma l'export CSV completo "
                "elenca anche quelle a zero.",
            ),
        ),
    ),
    LegendSection(
        "Flag (colonna Flag)",
        (
            _e(
                FLAG_VALUE,
                "EV ≥ soglia: la selezione è di valore; lo stake è di norma > 0 ma può arrotondare a 0 (sotto 0,1 unità) sulle "
                "quote molto alte, dove il Kelly frazionario chiede una puntata minuscola. È il flag che si cerca.",
            ),
            _e(
                FLAG_REGRESSION_TRAP,
                "la selezione è Over o GG contro una squadra in «trappola di regressione»: nelle ultime 3 gare ha subito "
                "al massimo 1 gol (GA ≤ 1) ma gli avversari hanno prodotto almeno 3.5 xG cumulati (xGA ≥ 3.5). La difesa "
                "sta rendendo oltre il lecito e il mercato, che guarda i gol subiti, sottoprezza i gol contro di lei. Come "
                "usarla: l'Over e il GG contro quella squadra sono il lato giusto; i suoi Under / No Gol sono da evitare. "
                "Il flag è informativo: non cambia l'EV calcolato.",
            ),
            _e(
                f"{TRAP_FLAG_PREFIX}<squadra>",
                "accompagna REGRESSION_TRAP e dice quale squadra della partita è in trappola (possono essere entrambe).",
            ),
            _e(
                FLAG_LONGSHOT,
                "quota book sopra 6: anche con EV positivo è una scommessa che perde la gran parte delle volte, con "
                "lunghe serie negative. Il Kelly frazionario la dimensiona già piccola; sapere che lo è aiuta a non "
                "demoralizzarsi (o a saltarla, se la varianza non piace).",
            ),
            _e(
                FLAG_SUSPICIOUS_EDGE,
                "EV sopra il 25%: un vantaggio così grande sui bookmaker è quasi sempre un errore di dati o una quota "
                "stantia (partita rinviata, formazione nota al book e non al modello, quota pre-apertura, nome squadra "
                "sbagliato), non una miniera d'oro. Verifica la quota sul sito del book e la data della partita prima di "
                "puntare.",
            ),
        ),
    ),
    LegendSection(
        "Selezioni di giornata (tab Selezioni di giornata)",
        (
            _e(
                "Giornata",
                "la giornata corrente di ogni competizione della sezione scelta (i tre gironi di Serie C, la Serie B, i nove "
                "gironi di Serie D o un campionato abilitato), messe insieme qualunque sia il loro numero (gironi di Serie C alla 13ª, D-A alla 12ª) "
                "e con le partite già giocate incluse; le giornate successive raggruppano i turni seguenti giocati nello stesso "
                "blocco della settimana calcistica: fine settimana (venerdì-lunedì) oppure turno infrasettimanale "
                "(martedì-giovedì). Una competizione che in quel blocco non gioca (Serie B in sosta per le nazionali) manca "
                "dall'etichetta. Le liste X e GG considerano solo le partite ancora da giocare (calcio d'inizio non ancora "
                "passato: una partita iniziata esce dalle liste e dal pannello valore); rinvii e recuperi fuori turno "
                "non spostano la giornata (sono in «Recuperi e rinvii»). Ogni sezione raggruppa le giornate dal proprio "
                "calendario: le 34 giornate di Serie D non spostano quelle di Serie C o di Serie B.",
            ),
            _e(
                "Le 5 partite più probabili da X",
                "fra tutte le partite della giornata della sezione scelta (i tre gironi di Serie C insieme, la Serie B, oppure "
                "i nove gironi di Serie D insieme), le cinque con la probabilità "
                "di pareggio più alta secondo il modello. Ordinate **solo per probabilità**, non per valore: una X al 33% "
                "a quota 2.90 (EV −4%) sta in lista anche se non conviene. Nota «lambda e mu vicini» = squadre equilibrate, "
                "«pochi gol attesi» = gara chiusa: entrambe spingono il pareggio.",
            ),
            _e(
                "Le 12 partite più probabili da GG",
                "le dodici partite della giornata, su tutte le competizioni della sezione, con la probabilità più alta che entrambe "
                "le squadre segnino. Stesso criterio: solo probabilità del modello. Nota «xG attese alte» = tanti gol "
                "attesi in totale, «entrambi gli attacchi sopra 1.2 gol attesi» = nessuna delle due dovrebbe restare a "
                "secco.",
            ),
            _e(
                "«Più probabile» non è «conveniente»",
                "la probabilità dice quante volte l'esito accade, la quota dice quanto paga: un GG al 65% pagato 1.45 "
                "(fair 1.54) perde soldi nel lungo periodo, un GG al 55% pagato 2.00 (fair 1.82) li guadagna. Le colonne "
                "Quota Fair, Quota Book ed EV sono lì per questo: incrocia sempre la lista con la quota e con l'EV prima "
                "di puntare, e dà la precedenza alle righe che compaiono anche nel pannello valore.",
            ),
            _e(
                "Nessuna quota in lista",
                "se Quota Book, Bookmaker ed EV sono vuoti nessun bookmaker nei dati quota quella selezione: la lista "
                "resta utile per sapere dove guardare, il prezzo va cercato a mano.",
            ),
        ),
    ),
    LegendSection(
        "Schedine (tab Schedine)",
        (
            _e(
                "Schedina X",
                "le 5 partite della giornata (5 di default, regolabile) con la probabilità di pareggio più alta secondo il modello, "
                "fra le sezioni scelte e solo fra le partite non ancora iniziate; una partita entra al massimo una volta per "
                "schedina. Stesso criterio della lista X: probabilità, non valore.",
            ),
            _e(
                "Schedina GG",
                "le 10 partite (default) con la probabilità più alta che entrambe le squadre segnino.",
            ),
            _e(
                "Schedina mista",
                "per ogni partita il segno più probabile fra 1, 2, GG e Over 2.5 secondo il modello, poi le 6 partite (da 3 a "
                "10) in cui quel segno è più probabile: integra le partite con la probabilità più alta per il loro segno. Con "
                "«Solo partite quotate» il segno si sceglie fra quelli che hanno una quota; «P minima» esclude i segni sotto la "
                "soglia.",
            ),
            _e(
                "Probabilità della schedina",
                "il prodotto delle probabilità delle singole partite, nell'ipotesi che le partite siano indipendenti (un'ipotesi, "
                "non un fatto: due gare dello stesso girone con lo stesso arbitro o lo stesso meteo non lo sono del tutto). "
                "Cinque X al 35% fanno 0,35⁵ ≈ 0,5%: una schedina su 190. Quota fair = 1 / probabilità.",
            ),
            _e(
                "Quota book ed EV della schedina",
                "la quota book è il prodotto delle migliori quote delle partite, solo quando tutte sono quotate; EV = probabilità × "
                "quota book − 1. Se una partita non ha quota, quota book ed EV restano vuoti.",
            ),
            _e(
                "Margine composto",
                "una multipla moltiplica il margine del bookmaker: con il 5% per partita, 5 partite cedono circa il 22% della "
                "vincita equa e 10 partite quasi il 39%. Per questo una schedina lunga ha quasi sempre EV negativo anche quando "
                "ogni partita è «probabile»: le singole di valore (tab Opportunità) sono il modo più efficiente di sfruttare un "
                "vantaggio.",
            ),
            _e(
                "Partite vincenti attese",
                "la somma delle probabilità delle partite: quante ne indovini in media. Con 10 GG al 62% sono circa 6, e la "
                "schedina intera si vince solo se sono 10.",
            ),
            _e(
                "Stake suggerito della schedina",
                "Kelly frazionario sulla schedina intera (la frazione della sidebar), con un tetto di 1 unità; «non consigliata» "
                "quando l'EV è negativo o manca una quota. È volutamente piccolo: la varianza di una multipla è enorme.",
            ),
            _e(
                "Track record delle schedine",
                "le stesse tre schedine ricostruite per ogni giornata passata con le probabilità fuori campione del backtest: "
                "giornate, schedine vinte contro quelle attese, percentuale di gambe vinte e ROI alle quote reali quando ci sono, "
                "con intervallo di confidenza al 95%.",
            ),
        ),
    ),
    LegendSection(
        "Affidabilità del modello (tab Affidabilità modello)",
        (
            _e(
                "Backtest walk-forward",
                "ogni giornata passata (blocco della settimana calcistica) è prevista da un modello stimato solo sulle partite "
                "giocate prima del suo calcio d'inizio, con lo stesso decadimento, la stessa prior e lo storico di produzione. "
                "Sono le probabilità che il modello avrebbe pubblicato allora: la stima di produzione ha già visto quei risultati "
                "e non va mai usata per giudicare la precisione. Di default si valutano la stagione corrente e la precedente.",
            ),
            _e(
                "Log loss",
                "−ln della probabilità data all'esito avvenuto, in media: più bassa è meglio. Un modello che dà sempre 1/3 a 1, X "
                "e 2 fa 1,099; le frequenze di base di un campionato intorno a 1,07-1,09; un buon modello e i bookmaker fra 0,98 "
                "e 1,03 sull'1X2. Punisce molto le sicurezze sbagliate (un 90% che non esce costa più di dieci 55% sbagliati).",
            ),
            _e(
                "Brier",
                "errore quadratico medio fra probabilità ed esito (1X2 da 0 a 2, mercati a due esiti da 0 a 1): più basso è "
                "meglio, meno severo della log loss con gli errori rari.",
            ),
            _e(
                "Accuratezza (%)",
                "quante volte l'esito più probabile secondo il modello è avvenuto. Utile ma ingannevole: sull'1X2 il segno 1 "
                "vince spesso anche senza modello; conta che le probabilità siano giuste, non solo il favorito.",
            ),
            _e(
                "Base e Skill vs base",
                "la base sono le frequenze di 1/X/2, GG e Over del campionato nei 365 giorni precedenti (quello che sa chi non ha "
                "un modello). Skill vs base = 1 − log loss modello / log loss base: positiva = il modello batte le frequenze di "
                "base. È il primo esame da superare.",
            ),
            _e(
                "Mercato e Skill vs mercato",
                "le probabilità del bookmaker senza margine (quote pre-partita delle partite giocate, dallo storico FootyStats), "
                "valutate solo sulle «Partite con quote». Il modello va confrontato con il mercato sulle stesse partite: le "
                "colonne «Log loss modello (partite con quote)» e «Brier modello (partite con quote)» e il riquadro «Modello vs "
                "mercato»; «Log loss modello» copre tutte le partite valutate, anche quelle senza quote (la copertura cambia da "
                "campionato a campionato), e non è confrontabile con il mercato. Skill vs mercato = 1 − log loss modello / log "
                "loss mercato sulle stesse partite: battere il mercato è raro; un valore negativo di poco è normale e non "
                "impedisce di trovare singole di valore, uno molto negativo invita alla prudenza sugli EV.",
            ),
            _e(
                "Grafico di affidabilità",
                "le partite sono raggruppate per probabilità prevista (fasce di 10 punti): in orizzontale la probabilità media "
                "della fascia, in verticale quante volte l'esito è avvenuto, con la barra dell'intervallo di Wilson al 95%. "
                "Sulla diagonale il modello è tarato; sopra la diagonale è troppo prudente (l'esito accade più spesso), sotto è "
                "troppo sicuro. Una fascia con poche partite ha la barra lunga: non trarne conclusioni.",
            ),
            _e(
                "Hit rate, IC 95% e ROI",
                "hit rate = selezioni vinte / selezioni, con l'intervallo di Wilson al 95%; va confrontato con la «P media» del "
                "modello (se coincidono, il modello è tarato). ROI = profitto / puntate a puntata fissa sulla miglior quota "
                "pre-partita, con un intervallo al 95% (approssimazione normale): finché l'intervallo contiene lo zero il "
                "guadagno non si distingue dalla fortuna.",
            ),
            _e(
                "Multiple top-k delle liste",
                "le prime k partite (da 2 a 5) della lista X o GG di ogni giornata passata giocate insieme, per sezione o per "
                "tutte le sezioni: è la giocata «le prime 2 GG della Serie C» ripetuta su ogni giornata. Giornate, multiple "
                "vinte contro quelle attese (somma dei prodotti delle probabilità, indipendenza ipotizzata), win rate con "
                "intervallo di Wilson e ROI alle quote reali; il dettaglio giornata per giornata è in «Liste giornata per giornata».",
            ),
            _e(
                "Poche vincite non sono una prova",
                "una domenica con la schedina vinta (100 € con 5 €) o due GG azzeccati capitano anche senza alcun vantaggio: "
                "contano il numero di giornate e di partite valutate e gli intervalli di confidenza, non l'aneddoto.",
            ),
            _e(
                "Ricalibrazione validata",
                "imparare dal differenziale fra previsioni ed esiti: per ogni mercato una piccola correzione (Platt su GG/NG e "
                "Over/Under, temperatura multinomiale sull'1X2, con penalità verso «nessuna correzione»; parametri propri per "
                "Serie B, C o D solo con abbastanza partite) è stimata sulle giornate più vecchie del backtest e giudicata su "
                "quelle più recenti. Si applica solo ai mercati in cui la log loss fuori campione migliora almeno dello 0,5%; "
                "altrimenti restano le probabilità originali. «Calibrazione applicata: sì/no, miglioramento log loss X%» lo dice; "
                "l'interruttore «Usa probabilità ricalibrate» la accende o la spegne (è reversibile: il modello non viene "
                "toccato) e la tabella Giornata mostra accanto la probabilità originale («orig.»).",
            ),
            _e(
                "Ottimizzazione di xi e prior",
                "facoltativa: ripete il backtest della stagione corrente per 9 combinazioni di decadimento (ξ 0,001 / 0,0018 / "
                "0,003) e forza della prior. La combinazione è scelta sulle giornate più vecchie e confermata sul 30% più "
                "recente, che la scelta non ha visto: si può applicare con l'interruttore dedicato solo se lì batte la "
                "configurazione standard di almeno lo 0,5% (la soglia della ricalibrazione), altrimenti il guadagno è rumore. "
                "Quando è applicata, la tab Affidabilità avvisa che le giornate su cui è stata scelta non sono più del tutto "
                "fuori campione. Si torna indietro spegnendo l'interruttore.",
            ),
        ),
    ),
    LegendSection(
        "Dati e limiti",
        (
            _e(
                "Dati simulati vs reali",
                "la dashboard mostra solo dati reali FootyStats, scaricati con la tua chiave: il badge della sidebar dice "
                "«Dati reali FootyStats». Senza dati non mostra numeri ma una pagina iniziale che spiega cosa manca e come "
                "configurare la chiave. La lega simulata (squadre vere, calendari, risultati e quote del bookmaker fittizio "
                "SimBook inventati) esiste solo come demo esplicita, avviata con SERIEC_DEMO=1: allora ogni pagina ha in cima "
                "l'avviso rosso «DATI SIMULATI: squadre, calendari e risultati inventati. Non usare per scommettere.» e il badge "
                "dice «Demo»; ogni CSV scaricato dalla demo si chiama DEMO_... e ha come prima colonna AVVISO = «DATI SIMULATI - "
                "NON USARE PER SCOMMETTERE». **Non usare mai la demo per scommettere.**",
            ),
            _e(
                "Aggiornamento dei dati FootyStats",
                "all'avvio, se la chiave è configurata e i dati locali mancano o hanno più di 6 ore, la dashboard li scarica di "
                "nuovo da FootyStats (partite delle ultime 3 stagioni, calendario con orari e rinvii, quote) e allinea il "
                "dizionario squadre; la sidebar dice «Dati FootyStats aggiornati alle HH:MM del gg/mm» e il pulsante «Aggiorna "
                "dati ora» li riscarica subito. Una competizione che non si può scaricare (lega non scelta sull'account, errore "
                "di rete, anche solo per la stagione in corso) è segnalata con un avviso e tiene i dati precedenti; le altre si "
                "aggiornano. Finché i dati restano più vecchi di 6 ore (chiave mancante, aggiornamento non riuscito) un avviso "
                "giallo in cima alla pagina ne dice l'età e il motivo, e un altro avviso resta per ogni competizione ferma ai "
                "dati di un download precedente. Le differenze del dizionario squadre sono in «Dettagli pipeline».",
            ),
            _e(
                "Chiave FootyStats e dati locali",
                "la chiave non va mai pubblicata: in locale sta nel file .env (FOOTYSTATS_API_KEY=...), su Streamlit Cloud in "
                "App settings → Secrets (FOOTYSTATS_API_KEY = \"...\"). I dati scaricati restano solo in data/processed e "
                "data/odds sul tuo computer o sul server dell'app: non sono nel repository pubblico, perché i dati FootyStats "
                "sono in licenza al tuo account.",
            ),
            _e(
                "Storico di 3 stagioni",
                "il modello è stimato sulla stagione in corso più le due precedenti; le partite vecchie contano meno "
                "(ξ). Le squadre presenti solo nello storico (retrocesse, promosse, escluse) restano nella stima come "
                "avversarie ma non compaiono in classifica, previsioni e valore.",
            ),
            _e(
                "Squadre storiche e perimetro",
                "il perimetro è il dizionario squadre della stagione (config/teams_<stagione>.json), allineato alle liste "
                "squadre di FootyStats a ogni aggiornamento: le 80 squadre di Serie C A/B/C e Serie B più le 162 dei nove "
                "gironi di Serie D. Una partita in programma "
                "con una squadra fuori perimetro o datata in un'altra stagione viene saltata e segnalata in «Dettagli pipeline».",
            ),
            _e(
                "Sezioni",
                "il selettore «Sezione» della sidebar ha una pagina per campionato: Serie C (gironi A, B, C), Serie B, Serie D "
                "(gironi A-I) e ogni campionato abilitato in config/competitions.toml. La stima Dixon-Coles è una per paese: "
                "Serie B, C e D sono stimate insieme (le squadre promosse e retrocesse collegano le divisioni; fattore campo e "
                "Σα = 0 per competizione tengono i gironi confrontabili), un campionato estero a parte. Classifica, giornate, "
                "risultati, valore, selezioni ed export mostrano solo la sezione scelta; le schedine e l'affidabilità possono "
                "unire più sezioni.",
            ),
            _e(
                "Abilitare Serie A o un campionato europeo",
                "Serie A, Premier League, La Liga, Bundesliga, Ligue 1, Eredivisie e Primeira Liga sono già configurati in "
                "config/competitions.toml con enabled = false: per accenderne uno metti enabled = true, sceglilo sull'account "
                "footystats.org («Choose leagues», il piano limita il numero di leghe) e premi «Aggiorna dati ora». La raccolta "
                "riempie le sue squadre nel dizionario e ne scarica partite, calendario e quote; il campionato diventa una sezione "
                "nuova (la Serie A entra nella stima italiana, un campionato estero ha la sua). enabled = false lo spegne di nuovo.",
            ),
            _e(
                "Serie D (sezione dedicata)",
                "nove gironi (A-I) di 18 squadre, 34 giornate ciascuno, codici DA-DI, nella stessa stima di Serie B e Serie C; "
                "classifica, giornate, risultati, valore, selezioni ed export mostrano solo la sezione e le liste X / GG sono "
                "calcolate dentro la sezione. Le liste squadre dei gironi vengono da FootyStats e si riallineano a ogni "
                "aggiornamento dei dati; finché non ci sono partite di Serie D nei dati la sezione mostra solo un avviso con il "
                "comando di raccolta (python -m src.ingestion.footystats --competitions DA DB DC DD DE DF DG DH DI --seasons 3). "
                "Le statistiche di "
                "Serie D su FootyStats sono più povere: xG e tiri spesso mancano (colonne «-», λ e μ restano stimati dai gol), "
                "senza xG il filtro REGRESSION_TRAP non può scattare, e le quote coprono meno partite (pannello valore in parte "
                "vuoto): leggi gli EV di Serie D con prudenza ancora maggiore.",
            ),
            _e(
                "Copertura minima",
                "sotto le 15 partite giocate per squadra nella competizione meno coperta la prior resta attiva; sotto le "
                "5-6 le stime sono comunque fragili e gli EV vanno letti con sospetto (le quote dei book incorporano già "
                "informazioni che il modello non ha ancora visto).",
            ),
            _e(
                "Prime giornate",
                "nelle prime 3-4 giornate il modello si regge quasi solo sullo storico e sulla prior: usalo per orientarti "
                "(gol attesi, squadre equilibrate), punta poco o niente sulle sole indicazioni di valore e aspetta che la "
                "copertura cresca. Il filtro REGRESSION_TRAP ha bisogno di 3 gare giocate per squadra.",
            ),
            _e(
                "Quote e tempi",
                "le quote cambiano fino al calcio d'inizio: un EV calcolato su quote di ieri può non esistere più oggi. "
                "Confronta la quota presa con quella di chiusura (closing line): batterla stabilmente è il segnale più "
                "affidabile che il modello vede qualcosa.",
            ),
        ),
    ),
    LegendSection(
        "Regole di buon senso",
        (
            _e(
                "Bankroll fisso e separato",
                "decidi all'inizio quanto dedicare alle scommesse, dividilo in 100 unità e dimensiona tutto in unità; "
                "non ricaricare a stagione in corso e non puntare mai più del tappo per selezione.",
            ),
            _e(
                "Mai inseguire le perdite",
                "una serie negativa è normale anche con EV positivo (con quote intorno a 2.70 si perde il 60% delle "
                "volte); raddoppiare la puntata per rifarsi distrugge il vantaggio del Kelly frazionario.",
            ),
            _e(
                "Campione minimo prima di fidarsi",
                "qualche decina di scommesse non dice nulla: servono almeno 200-300 selezioni registrate (quota presa, "
                "EV, esito) per distinguere la fortuna dal valore. Tieni un registro.",
            ),
            _e(
                "Confronta con le quote di chiusura",
                "se le quote che prendi sono in media più alte di quelle di chiusura il modello anticipa il mercato; se "
                "sono più basse stai pagando troppo, qualunque cosa dica l'EV.",
            ),
        ),
    ),
)


def legend_sections() -> tuple[LegendSection, ...]:
    """The legend sections, in display order (same object as :data:`LEGEND`)."""
    return LEGEND


def legend_terms() -> dict[str, str]:
    """``term -> meaning`` for every entry of the legend (a flat lookup for tooltips and tests)."""
    return {entry.term: entry.meaning for section in LEGEND for entry in section.entries}


def legend_markdown() -> str:
    """The whole legend as Markdown: ``## <LEGEND_TITLE>``, one ``### `` heading and bullet list per section.

    Used verbatim by the dashboard's *Legenda* tab (``st.markdown``) and pasted
    into ``README.md`` between the ``<!-- legend:start -->`` /
    ``<!-- legend:end -->`` markers; ``tests/test_picks.py`` checks that the
    README copy is up to date.
    """
    parts = [
        f"## {LEGEND_TITLE}",
        "",
        (
            "Glossario di tutti i numeri mostrati da dashboard, console ed export, scritto per chi li usa per pianificare "
            "le scommesse. Le formule sono quelle del codice (`src/engine/model.py`, `src/engine/value.py`, "
            "`src/engine/picks.py`, `src/engine/tables.py`)."
        ),
    ]
    for section in LEGEND:
        parts += ["", section.to_markdown()]
    return "\n".join(parts) + "\n"


# =============================================================================
# Worked example
# =============================================================================
#: Flags that make a value row a poor teaching example (a long shot or a probable data error).
POOR_EXAMPLE_FLAGS: Final[frozenset[str]] = frozenset({FLAG_LONGSHOT, FLAG_SUSPICIOUS_EDGE})


def pick_example_row(rows: Sequence[ValueOpportunity]) -> ValueOpportunity | None:
    """The value-panel row to explain: the first one without LONGSHOT / SUSPICIOUS_EDGE, else the first, ``None`` for no rows.

    ``rows`` are the panel's rows in display order (EV descending), so the
    pick is the best-EV *ordinary* row: a long shot or an EV above 25% is, by
    the legend's own words, a probable data error and the worst row to teach
    the reading of a value line with.
    """
    for row in rows:
        if POOR_EXAMPLE_FLAGS.isdisjoint(row.flags):
            return row
    return rows[0] if rows else None


@dataclass(frozen=True)
class WorkedExample:
    """Numbers of one value-panel row, used by :func:`worked_example_markdown`.

    Fractions, not percentages (``p_model=0.40``, ``ev=0.08``); ``bankroll_units``
    and ``kelly_fraction`` are the staking parameters the stake was computed
    with, so the explanation reproduces the row's own numbers.
    """

    home_team: str
    away_team: str
    competition: str
    market: str
    selection: str
    bookmaker: str
    book_odds: float
    p_model: float
    p_implied_raw: float
    p_implied_devig: float
    kelly_fraction: float = KELLY_EIGHTH
    bankroll_units: float = 100.0
    flags: tuple[str, ...] = ()
    is_synthetic: bool = False
    synthetic_reason: str = SYNTHETIC_REASON_NO_VALUE

    def __post_init__(self) -> None:
        if not 0.0 < self.p_model < 1.0:
            raise ValueError(f"p_model must be in (0, 1), got {self.p_model}")
        if self.book_odds <= 1.0:
            raise ValueError(f"book_odds must be > 1, got {self.book_odds}")
        if not 0.0 < self.kelly_fraction <= 1.0:
            raise ValueError(f"kelly_fraction must be in (0, 1], got {self.kelly_fraction}")
        if self.bankroll_units <= 0.0:
            raise ValueError("bankroll_units must be > 0")
        object.__setattr__(self, "flags", tuple(str(f) for f in self.flags))
        if self.is_synthetic and not self.synthetic_reason.strip():
            raise ValueError("a synthetic WorkedExample needs a reason")

    @classmethod
    def from_opportunity(cls, o: ValueOpportunity, *, bankroll_units: float = 100.0) -> WorkedExample:
        """The numbers of a real :class:`~src.engine.value.ValueOpportunity` row."""
        return cls(
            home_team=o.home_team,
            away_team=o.away_team,
            competition=o.girone.short,
            market=o.market.value,
            selection=o.selection,
            bookmaker=o.bookmaker,
            book_odds=float(o.book_odds),
            p_model=float(o.p_model),
            p_implied_raw=float(o.p_implied_raw),
            p_implied_devig=float(o.p_implied_devig),
            kelly_fraction=float(o.kelly_fraction),
            bankroll_units=float(bankroll_units),
            flags=tuple(o.flags),
        )

    @classmethod
    def synthetic(
        cls,
        *,
        kelly_fraction: float = KELLY_EIGHTH,
        bankroll_units: float = 100.0,
        reason: str = SYNTHETIC_REASON_NO_VALUE,
    ) -> WorkedExample:
        """A made-up but realistic row (home win at 2.70 with the model at 40%) when no real row can be shown.

        ``reason`` says why, in the header of :func:`worked_example_markdown`:
        :data:`SYNTHETIC_REASON_NO_QUOTES` when the state has no bookmaker quote at
        all, :data:`SYNTHETIC_REASON_NO_VALUE` (default) when quotes exist but no
        selection clears the current filters (matchday, competition, EV threshold).
        """
        return cls(
            home_team="Squadra Casa",
            away_team="Squadra Ospite",
            competition="Girone A",
            market="1X2",
            selection="1",
            bookmaker="BookEsempio",
            book_odds=2.70,
            p_model=0.40,
            p_implied_raw=1.0 / 2.70,
            p_implied_devig=0.3527,
            kelly_fraction=kelly_fraction,
            bankroll_units=bankroll_units,
            flags=(FLAG_VALUE,),
            is_synthetic=True,
            synthetic_reason=reason,
        )

    @property
    def fair_odds(self) -> float:
        return 1.0 / self.p_model

    @property
    def ev(self) -> float:
        return self.p_model * self.book_odds - 1.0

    @property
    def edge(self) -> float:
        return self.p_model - self.p_implied_devig

    @property
    def kelly_full(self) -> float:
        return kelly_fraction(self.p_model, self.book_odds)

    @property
    def stake(self) -> float:
        return stake_units(self.p_model, self.book_odds, KellyConfig(fraction=self.kelly_fraction, bankroll_units=self.bankroll_units))


def _fraction_label(fraction: float) -> str:
    """``'1/8'`` for 0.125, ``'1/4'`` for 0.25, otherwise the decimal."""
    for denominator in (2, 3, 4, 5, 8, 10, 16, 20):
        if abs(fraction - 1.0 / denominator) < 1e-9:
            return f"1/{denominator}"
    return f"{fraction:g}"


def worked_example_markdown(example: WorkedExample) -> str:
    """Step-by-step Italian reading of one value-panel row, with the example's own numbers.

    Fair odds, EV, edge, full Kelly and the fractional stake are recomputed from
    ``p_model``, ``book_odds`` and the de-vigged implied probability with the
    same functions the value engine uses, so the explanation never drifts from
    the table.
    """
    p = example.p_model
    b = example.book_odds - 1.0
    q = 1.0 - p
    fair = example.fair_odds
    ev = example.ev
    edge = example.edge
    f_star = example.kelly_full
    stake = example.stake
    fraction = _fraction_label(example.kelly_fraction)
    header = (
        f"**{example.home_team} - {example.away_team}** ({example.competition}) · mercato {example.market}, esito "
        f"**{example.selection}** · quota {example.book_odds:.2f} di {example.bookmaker}"
    )
    if example.is_synthetic:
        header += f"  \n*Esempio costruito: {example.synthetic_reason}; i numeri sono inventati ma realistici.*"
    verdict = (
        f"la quota book ({example.book_odds:.2f}) è sopra la fair ({fair:.2f}): il book paga più di quanto il modello "
        f"ritenga giusto, la selezione ha valore."
        if ev > 0
        else f"la quota book ({example.book_odds:.2f}) è sotto la fair ({fair:.2f}): il book paga meno di quanto il "
        f"modello ritenga giusto, niente valore."
    )
    lines = [
        header,
        "",
        f"1. **Probabilità del modello**: P = {p:.1%}. **Quota fair** = 1 / P = 1 / {p:.3f} = **{fair:.2f}**.",
        (
            f"2. **Probabilità implicita** della quota: 1 / {example.book_odds:.2f} = {example.p_implied_raw:.1%}; dopo il "
            f"de-vig (tolto il margine del book) **{example.p_implied_devig:.1%}**."
        ),
        f"3. **EV** = P × Quota − 1 = {p:.3f} × {example.book_odds:.2f} − 1 = **{ev:+.1%}**: {verdict}",
        (
            f"4. **Edge** = P_modello − P_implicita de-viggata = {p:.1%} − {example.p_implied_devig:.1%} = **{edge:+.1%}** "
            f"(il modello è {'più' if edge >= 0 else 'meno'} ottimista del book di {abs(edge):.1%})."
        ),
        (
            f"5. **Kelly pieno** f* = (p·b − q) / b con b = quota − 1 = {b:.2f} e q = 1 − p = {q:.3f}: "
            f"({p:.3f} × {b:.2f} − {q:.3f}) / {b:.2f} = **{f_star:.1%}** del bankroll"
            + (" (zero: EV negativo, non si punta)." if f_star <= 0 else ".")
        ),
        (
            f"6. **Stake** = f* × {fraction} × {example.bankroll_units:g} unità = {f_star:.3f} × {example.kelly_fraction:g} × "
            f"{example.bankroll_units:g} = **{stake:.1f} unità** (arrotondato a 0,1; tappo 3 unità; sotto 0,1 ⇒ 0)."
        ),
    ]
    if example.flags:
        notes = {
            FLAG_VALUE: "EV sopra soglia, la riga compare nel pannello",
            FLAG_REGRESSION_TRAP: "Over/GG contro una difesa in sovra-rendimento: il lato giusto",
            FLAG_LONGSHOT: "quota sopra 6, perde spesso anche se ha valore",
            FLAG_SUSPICIOUS_EDGE: "EV sopra il 25%: controlla la quota prima di puntare",
        }
        described = [f"`{flag}` ({notes[flag]})" if flag in notes else f"`{flag}`" for flag in example.flags]
        lines.append(f"7. **Flag**: {', '.join(described)}.")
    else:
        lines.append("7. **Flag**: nessuno (EV sotto soglia, nessuna trappola, quota normale).")
    return "\n".join(lines) + "\n"
