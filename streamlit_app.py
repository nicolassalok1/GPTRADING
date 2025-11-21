import streamlit as st
from openai import OpenAI
import os
import json
import alpaca_trade_api as tradeapi
import time
from dotenv import load_dotenv
import pandas as pd

# Configuration
DATA_FILE = "equities.json"
PORTFOLIO_FILE = "portfolio.json"
load_dotenv()

# Alpaca API Setup
key = "PKRQ4GPVDAPCYIH6QGR4HI5USK"
secret_key = "3mENa9jXaLhESSekQzvz4cRh758awvBppB7Dfs9o1LJw"
BASE_URL = "https://paper-api.alpaca.markets/"
api = tradeapi.REST(key, secret_key, BASE_URL, api_version="v2")

# Page config
st.set_page_config(page_title="AI Trading Bot", page_icon="📈", layout="wide")

# Helper functions
@st.cache_data(ttl=10)
def fetch_portfolio():
    try:
        positions = api.list_positions()
        portfolio = []
        for pos in positions:
            portfolio.append({
                'Symbol': pos.symbol,
                'Quantity': pos.qty,
                'Entry Price': f"${float(pos.avg_entry_price):.2f}",
                'Current Price': f"${float(pos.current_price):.2f}",
                'Unrealized P/L': f"${float(pos.unrealized_pl):.2f}",
                'Side': 'buy'
            })
        return portfolio
    except Exception as e:
        st.error(f"Error fetching portfolio: {e}")
        return []

@st.cache_data(ttl=10)
def fetch_open_orders():
    try:
        orders = api.list_orders(status='open')
        open_orders = []
        for order in orders:
            open_orders.append({
                'Symbol': order.symbol,
                'Quantity': order.qty,
                'Limit Price': f"${float(order.limit_price):.2f}" if order.limit_price else "Market",
                'Side': order.side
            })
        return open_orders
    except Exception as e:
        st.error(f"Error fetching orders: {e}")
        return []

def get_data(symbol):
    try:
        barset = api.get_latest_trade(symbol)
        return {"price": barset.price}
    except Exception as e:
        return {"price": -1}

def load_equities():
    try:
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_equities(equities):
    with open(DATA_FILE, 'w') as f:
        json.dump(equities, f, indent=2)

def load_portfolio():
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)

def buy_asset(symbol, quantity, price):
    portfolio = load_portfolio()
    if symbol in portfolio:
        old_qty = portfolio[symbol]['quantity']
        old_avg = portfolio[symbol]['avg_price']
        new_qty = old_qty + quantity
        new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
        portfolio[symbol] = {
            'quantity': new_qty,
            'avg_price': round(new_avg, 2),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    else:
        portfolio[symbol] = {
            'quantity': quantity,
            'avg_price': price,
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    save_portfolio(portfolio)
    return portfolio[symbol]

def sell_asset(symbol, quantity):
    portfolio = load_portfolio()
    if symbol in portfolio:
        current_qty = portfolio[symbol]['quantity']
        if quantity >= current_qty:
            del portfolio[symbol]
        else:
            portfolio[symbol]['quantity'] = current_qty - quantity
            portfolio[symbol]['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save_portfolio(portfolio)
        return True
    return False

def chatgpt_response(message: str):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        portfolio_data = json.dumps(fetch_portfolio(), indent=2)
        open_orders = json.dumps(fetch_open_orders(), indent=2)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI Portfolio Manager specializing in portfolio analysis, "
                    "risk management, and strategic market insights. "
                    "Your goal is to provide data-driven evaluations and professional recommendations."
                )
            },
            {
                "role": "user",
                "content": f"""
                    Here is my current portfolio:
                    {portfolio_data}

                    Here are my open orders:
                    {open_orders}

                    Your tasks:
                    1. Evaluate the risk exposures of my current holdings.
                    2. Analyze the potential impact of open orders.
                    3. Provide insights into portfolio health, diversification, and trade adjustments.
                    4. Speculate on market outlook given current conditions.
                    5. Identify potential risks and suggest mitigation strategies.

                    Finally, answer this specific question:
                    {message}
                    """
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_max_entry_price(symbol):
    try:
        orders = api.list_orders(status="filled", limit=50)
        prices = [float(order.filled_avg_price) for order in orders if order.filled_avg_price and order.symbol == symbol]
        return max(prices) if prices else -1
    except Exception as e:
        st.error(f"Error fetching orders: {e}")
        return 0

def place_initial_order(symbol):
    try:
        api.submit_order(
            symbol=symbol,
            qty=1,
            side="buy",
            type="market",
            time_in_force="gtc"
        )
        st.success(f"Initial order placed for {symbol}")
        time.sleep(2)
        return True
    except Exception as e:
        st.error(f"Error placing initial order: {e}")
        return False

def place_limit_order(symbol, price):
    try:
        api.submit_order(
            symbol=symbol,
            qty=1,
            side='buy',
            type='limit',
            time_in_force='gtc',
            limit_price=price
        )
        st.success(f"Placed limit order for {symbol} @ ${price}")
        return True
    except Exception as e:
        st.error(f"Error placing order: {e}")
        return False

# Streamlit UI
st.title("📈 AI Trading Bot")
st.markdown("---")

with st.expander("📘 Tutoriel d'utilisation de l'outil"):
    st.markdown("""
    ### 📘 Mise en situation : ce que vous essayez de faire
    
    Imaginez que vous êtes un investisseur particulier qui en a assez de trader "au feeling" : 
    vous voulez arrêter de courir après le marché, structurer vos décisions et savoir exactement 
    pourquoi vous entrez, renforcez ou sortez d'une position.
    
    Cet outil est là pour vous aider à **transformer votre trading en processus**. 
    Vous cherchez principalement à optimiser trois choses :
    
    - **Votre risque** : limiter la profondeur des pertes (drawdown) que vous êtes prêt à accepter
    - **Votre prix moyen d'entrée** : profiter des baisses pour améliorer vos points d'achat au lieu de paniquer
    - **Votre temps et votre charge mentale** : automatiser ce qui peut l'être, et garder votre énergie pour les décisions importantes
    
    L'idée n'est pas de prédire le futur, mais de mettre un cadre autour de votre comportement : 
    ouvrir l'application, voir en quelques secondes si tout est sous contrôle, puis décider calmement 
    s'il y a une action à prendre aujourd'hui ou non.
    
    ---
    
    ### 🧭 Comment lire l'application
    
    L'application est construite comme un **parcours logique de trader discipliné** :
    
    1. Vous **observez** votre situation (ce que vous possédez, comment ça évolue, ce que vos systèmes ont fait)
    2. Vous **décidez** où mettre du capital, ce que vous voulez renforcer ou alléger
    3. Vous **exécutez** des ordres manuels quand vous voulez intervenir directement
    4. Vous **automatisez** certaines parties de votre stratégie avec des systèmes basés sur le drawdown et des niveaux de prix
    5. Vous **analysez** avec l'IA pour challenger vos idées, comprendre vos risques et clarifier votre stratégie
    
    Chaque onglet correspond à une étape de ce parcours. 
    Dans les sections "📚 Comment utiliser ..." au bas de chaque onglet, 
    vous trouverez une explication détaillée de **ce que vous êtes en train d'y faire** 
    (qu'est-ce que vous optimisez, quels sont les enjeux, où se situe la difficulté mentale).
    
    ---
    
    ### 🎯 Comment bien démarrer
    
    Pour une première utilisation, vous pouvez suivre ce mini-scénario :
    
    1. Ouvrez l'application et prenez un moment pour comprendre que le but n'est pas de "trader plus", 
       mais de **trader mieux, avec un plan**.
    2. Allez ensuite dans chaque onglet, l'un après l'autre, sans forcément passer d'ordres au début :
       contentez-vous de lire le texte d'aide en bas de page et de repérer les boutons qui déclenchent de vraies actions.
    3. Quand vous vous sentez à l'aise, commencez petit : 
       un premier achat manuel, un premier système Add Equity, une première question à l'IA.
    4. Revenez quelques jours plus tard pour voir ce qui s'est passé : 
       avez-vous respecté votre plan ? vos systèmes ont-ils réagi comme prévu ? qu'avez-vous appris ?
    
    Utilisé de cette manière, cet outil devient **un cadre d'apprentissage et d'optimisation** : 
    à chaque utilisation, vous comprenez un peu mieux votre propre comportement de trader, 
    et vous ajustez votre manière d'investir pour qu'elle soit plus cohérente, plus sereine et plus alignée avec votre tolérance au risque.
    """)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("Configure your trading bot parameters")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "💰 Buy/Sell", "➕ Add Equity", "🤖 AI Assistant", "📋 Trading Systems"])

# Tab 1: Dashboard
with tab1:
    with st.expander("📘 Comprendre le Dashboard"):
        st.markdown("""
        ### 📊 Ce que vous faites dans le Dashboard
        
        Dans cet onglet, vous vérifiez **où en est votre argent** en un coup d'œil.
        L'idée est simple : avant de prendre une décision, vous regardez la photo globale de votre portfolio.
        
        Le tableau *My Portfolio* vous montre chaque ligne : symbole, quantité, prix moyen payé,
        prix actuel, valeur de la position et P&L. C'est ici que vous voyez immédiatement
        quelles positions tirent le portfolio vers le haut ou vers le bas.
        
        La métrique *Total Portfolio Value* vous donne la taille totale de votre capital investi.
        Vous pouvez l'utiliser comme point de repère jour après jour pour suivre l'évolution globale.
        
        La section *Configured Trading Systems* vous rappelle en un clin d'œil
        quels systèmes automatiques sont en place, sur quels actifs, avec quel drawdown
        et combien de niveaux sont définis. Vous voyez aussi rapidement leur statut (On/Off).
        
        La zone *Quick Remove* sert à faire du ménage : si un système ne vous convient plus,
        vous pouvez le supprimer en un clic sans aller dans d'autres onglets.
        
        La bonne pratique ici est de commencer chaque session par ce Dashboard :
        repérez les mouvements extrêmes, les positions trop lourdes ou les systèmes
        dont vous ne comprenez plus la logique, avant d'aller agir dans les autres onglets.
        """)
    st.subheader("💰 My Portfolio")
    my_portfolio = load_portfolio()
    
    if my_portfolio:
        portfolio_data = []
        total_value = 0
        for symbol, data in my_portfolio.items():
            current_price_data = get_data(symbol)
            current_price = current_price_data['price'] if current_price_data['price'] > 0 else data['avg_price']
            position_value = data['quantity'] * current_price
            pnl = (current_price - data['avg_price']) * data['quantity']
            pnl_pct = ((current_price - data['avg_price']) / data['avg_price'] * 100) if data['avg_price'] > 0 else 0
            total_value += position_value
            
            portfolio_data.append({
                'Symbol': symbol,
                'Quantity': data['quantity'],
                'Avg Price': f"${data['avg_price']:.2f}",
                'Current Price': f"${current_price:.2f}",
                'Value': f"${position_value:.2f}",
                'P&L': f"${pnl:.2f}",
                'P&L %': f"{pnl_pct:.2f}%"
            })
        
        df_my_portfolio = pd.DataFrame(portfolio_data)
        st.dataframe(df_my_portfolio, width="stretch", hide_index=True)
        st.metric("Total Portfolio Value", f"${total_value:.2f}")
    else:
        st.info("No assets in portfolio. Use the Buy/Sell tab to add positions.")
    
    # Trading Systems Section
    st.markdown("---")
    st.subheader("🎯 Configured Trading Systems")
    equities = load_equities()
    
    if equities:
        systems_data = []
        for symbol, data in equities.items():
            systems_data.append({
                'Symbol': symbol,
                'Position': data['position'],
                'Entry Price': f"${data['entry_price']:.2f}",
                'Drawdown': f"{data['drawdown']*100:.1f}%",
                'Levels': len(data['levels']),
                'Status': data['status']
            })
        
        df_systems = pd.DataFrame(systems_data)
        st.dataframe(df_systems, width="stretch", hide_index=True)
        
        # Quick remove section
        st.markdown("**Quick Remove:**")
        remove_cols = st.columns(len(equities) if len(equities) <= 5 else 5)
        for idx, symbol in enumerate(list(equities.keys())[:5]):
            with remove_cols[idx]:
                if st.button(f"🗑️ {symbol}", key=f"dash_remove_{symbol}"):
                    del equities[symbol]
                    save_equities(equities)
                    st.success(f"Removed {symbol}")
                    time.sleep(0.5)
                    st.rerun()
    else:
        st.info("No trading systems configured. Add an equity in the 'Add Equity' tab.")

# Tab 2: Buy/Sell
with tab2:
    with st.expander("📘 Comprendre Buy/Sell"):
        st.markdown("""
        ### 💰 Ce que vous faites dans Buy/Sell
        
        Cet onglet est votre **outil d'exécution manuelle** : c'est ici que vous décidez
        consciemment d'entrer ou de sortir d'une position, en contrôlant précisément prix et quantité.
        
        Le bloc *Buy Asset* vous sert à ouvrir ou renforcer une position.
        Vous indiquez le symbole, le nombre d'unités que vous voulez acheter,
        puis vous validez ou ajustez le prix proposé en fonction du marché.
        
        L'application calcule pour vous le **coût total** de l'opération,
        ce qui vous évite de faire les calculs de tête et vous aide à rester conscient
        du capital réellement engagé sur chaque trade.
        
        Le bloc *Sell Asset* est l'équivalent côté sorties : vous sélectionnez
        une position existante, choisissez combien d'unités vendre et,
        en fonction du prix de vente, vous voyez immédiatement le P&L associé.
        
        Utilisez cet onglet quand vous voulez **reprendre la main** sur une position :
        prendre vos profits, couper une perte, ou ajuster la taille d'un trade
        indépendamment de ce que font vos systèmes automatiques.
        
        La bonne habitude est de toujours regarder le P&L, le pourcentage
        et l'impact sur votre capital global avant de cliquer sur *Execute* :
        cela vous aide à prendre des décisions moins impulsives et plus alignées
        avec votre plan global de gestion du risque.
        """)
    st.subheader("💰 Buy/Sell Assets")
    
    col1, col2 = st.columns(2)
    
    # BUY Section
    with col1:
        st.markdown("### 📈 Buy Asset")
        buy_symbol = st.text_input("Symbol to Buy", placeholder="e.g., AAPL", key="buy_symbol").upper()
        buy_quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="buy_qty")
        
        if buy_symbol:
            price_data = get_data(buy_symbol)
            if price_data['price'] > 0:
                st.info(f"Current price: ${price_data['price']:.2f}")
                buy_price = st.number_input("Buy Price", min_value=0.01, value=float(price_data['price']), step=0.01, key="buy_price")
                total_cost = buy_quantity * buy_price
                st.metric("Total Cost", f"${total_cost:.2f}")
                
                if st.button("✅ Execute Buy", type="primary", key="exec_buy"):
                    result = buy_asset(buy_symbol, buy_quantity, buy_price)
                    st.success(f"Bought {buy_quantity} shares of {buy_symbol} @ ${buy_price:.2f}")
                    st.info(f"New position: {result['quantity']} shares @ avg ${result['avg_price']:.2f}")
                    time.sleep(1)
                    st.rerun()
            else:
                st.error(f"Could not fetch price for {buy_symbol}")
    
    # SELL Section
    with col2:
        st.markdown("### 📉 Sell Asset")
        my_portfolio = load_portfolio()
        
        if my_portfolio:
            sell_symbol = st.selectbox("Symbol to Sell", options=list(my_portfolio.keys()), key="sell_symbol")
            
            if sell_symbol:
                current_qty = my_portfolio[sell_symbol]['quantity']
                avg_price = my_portfolio[sell_symbol]['avg_price']
                
                st.info(f"Current holdings: {current_qty} shares @ avg ${avg_price:.2f}")
                
                sell_quantity = st.number_input("Quantity to Sell", min_value=1, max_value=int(current_qty), value=1, step=1, key="sell_qty")
                
                price_data = get_data(sell_symbol)
                if price_data['price'] > 0:
                    sell_price = st.number_input("Sell Price", min_value=0.01, value=float(price_data['price']), step=0.01, key="sell_price")
                    total_proceeds = sell_quantity * sell_price
                    pnl = (sell_price - avg_price) * sell_quantity
                    st.metric("Total Proceeds", f"${total_proceeds:.2f}")
                    st.metric("P&L", f"${pnl:.2f}", delta=f"{(pnl/total_proceeds*100):.2f}%")
                    
                    if st.button("✅ Execute Sell", type="primary", key="exec_sell"):
                        if sell_asset(sell_symbol, sell_quantity):
                            st.success(f"Sold {sell_quantity} shares of {sell_symbol} @ ${sell_price:.2f}")
                            st.info(f"P&L: ${pnl:.2f}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to execute sell order")
                else:
                    st.error(f"Could not fetch price for {sell_symbol}")
        else:
            st.info("No assets in portfolio to sell")

# Tab 3: Add Equity
with tab3:
    with st.expander("📘 Comprendre Add Equity"):
        st.markdown("""
        ### ➕ Ce que vous faites dans Add Equity
        
        Ici, vous ne passez pas d'ordres immédiats : vous **concevez des systèmes automatiques**
        qui vont acheter pour vous selon une logique de drawdown et de niveaux de prix.
        
        Le champ *Symbol* sert à choisir l'actif que vous voulez suivre
        de façon structurée (indice, action, ETF, crypto, etc.).
        
        *Number of Levels* définit combien de paliers d'achat vous voulez.
        Chaque niveau correspondra à un prix plus bas, où le système achètera
        automatiquement une unité supplémentaire lorsque le marché corrigera.
        
        *Drawdown %* contrôle l'écart entre ces niveaux de prix :
        un pourcentage faible donnera des niveaux proches les uns des autres,
        un pourcentage plus élevé espacera davantage les achats.
        
        Quand vous cliquez sur *Add Equity*, l'outil calcule les niveaux
        à partir du prix actuel et enregistre un système en mode *Off* par défaut,
        que vous pourrez ensuite activer et superviser dans l'onglet *Trading Systems*.
        
        Utilisez cet onglet pour **planifier à l'avance** comment vous voulez acheter
        pendant les baisses, plutôt que d'improviser dans la panique quand le marché chute.
        
        Avant de valider, demandez-vous toujours si vous êtes à l'aise
        avec le nombre de niveaux, le drawdown choisi et le capital total
        que cela représentera si tous les niveaux sont déclenchés.
        """)
    st.subheader("➕ Add New Equity")
    
    # Check current count
    equities = load_equities()
    current_count = len(equities)
    
    st.info(f"Trading Systems: {current_count}/10")
    
    if current_count >= 10:
        st.error("⚠️ Maximum limit reached! You cannot add more than 10 equities. Please remove one first.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.text_input("Symbol", placeholder="e.g., AAPL").upper()
        
        with col2:
            levels = st.number_input("Number of Levels", min_value=1, max_value=10, value=5)
        
        with col3:
            drawdown = st.number_input("Drawdown %", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
        
        if st.button("➕ Add Equity", type="primary"):
            if symbol:
                equities = load_equities()
                
                # Double check limit
                if len(equities) >= 10:
                    st.error("Maximum limit of 10 equities reached!")
                elif symbol in equities:
                    st.warning(f"{symbol} already exists!")
                else:
                    price_data = get_data(symbol)
                    entry_price = price_data['price']
                    
                    if entry_price > 0:
                        drawdown_decimal = drawdown / 100
                        level_prices = {str(i+1): round(entry_price * (1 - drawdown_decimal * (i+1)), 2) for i in range(levels)}
                        
                        equities[symbol] = {
                            "position": 0,
                            "entry_price": entry_price,
                            "levels": level_prices,
                            "drawdown": drawdown_decimal,
                            "status": "Off"
                        }
                        
                        save_equities(equities)
                        st.success(f"✅ Added {symbol} at ${entry_price:.2f}")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Could not fetch price for {symbol}")
            else:
                st.error("Please enter a symbol")

# Tab 4: AI Assistant
with tab4:
    with st.expander("📘 Comprendre l'AI Assistant"):
        st.markdown("""
        ### 🤖 Ce que vous faites dans AI Assistant
        
        Cet onglet transforme votre portfolio en **cas d'étude vivant** pour une IA spécialisée :
        vous lui posez des questions et elle vous répond à partir de vos données réelles.
        
        Le chat fonctionne comme une conversation : vos messages s'affichent à gauche,
        ceux de l'IA à droite, et l'historique est conservé tant que la session reste ouverte.
        
        Vous pouvez demander une analyse de vos positions, un avis
        sur votre niveau de diversification, ou un éclairage sur un risque spécifique.
        
        C'est aussi un espace pédagogique : si un concept vous échappe
        (drawdown, corrélation, volatilité, etc.), vous pouvez demander
        une explication contextualisée à partir de votre situation.
        
        La meilleure façon d'utiliser cet onglet est de **formuler vos doutes** :
        "Qu'est-ce qui pourrait mal se passer avec mon portfolio actuel ?",
        "Où suis-je trop exposé ?", "Qu'est-ce que je n'ai pas vu ?".
        
        Voyez l'IA comme un coach qui challenge vos idées,
        pas comme une boule de cristal. Plus vos questions sont claires,
        plus les réponses vous aideront à affiner votre propre réflexion.
        """)
    st.subheader("🤖 AI Portfolio Manager")
    st.markdown("Ask questions about your portfolio and get AI-powered insights")
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your portfolio..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatgpt_response(prompt)
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

# Tab 5: Trading Systems
with tab5:
    with st.expander("📘 Comprendre Trading Systems"):
        st.markdown("""
        ### 📋 Ce que vous faites dans Trading Systems
        
        Dans cet onglet, vous **pilotez vos systèmes automatiques en production** :
        c'est la salle de contrôle où vous vérifiez ce que vos robots sont en train de faire.
        
        Chaque bloc correspond à un actif pour lequel vous avez créé un système
        dans l'onglet *Add Equity*. Vous y voyez la position actuelle, le prix d'entrée,
        le drawdown configuré et le statut On/Off.
        
        Le toggle *Active* vous permet d'allumer ou d'éteindre un système en un clic.
        Quand il est sur *On*, le système surveille le marché en arrière-plan
        et place des ordres aux niveaux de prix que vous avez définis.
        
        Le tableau *Price Levels* récapitule tous les paliers d'achat prévus :
        c'est une façon visuelle de vérifier que la configuration correspond bien
        à votre intention initiale (espacement, nombre de niveaux, profondeur totale).
        
        Le bouton *Remove* supprime complètement un système quand vous ne voulez plus
        qu'il consomme de capital ou qu'il fasse partie de votre stratégie.
        
        Utilisez cet onglet pour faire des **revues régulières** de vos systèmes :
        vérifier qu'ils sont toujours pertinents, qu'ils ne se chevauchent pas trop,
        et qu'ils restent cohérents avec l'évolution de votre portefeuille global.
        """)
    st.subheader("📋 Trading Systems")
    
    equities = load_equities()
    
    if equities:
        for symbol, data in equities.items():
            with st.expander(f"{symbol} - Status: {data['status']}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Position", data['position'])
                
                with col2:
                    st.metric("Entry Price", f"${data['entry_price']:.2f}")
                
                with col3:
                    st.metric("Drawdown", f"{data['drawdown']*100:.1f}%")
                
                with col4:
                    current_status = data['status']
                    new_status = st.toggle(
                        "Active", 
                        value=current_status == "On",
                        key=f"toggle_{symbol}"
                    )
                    
                    if (new_status and current_status == "Off") or (not new_status and current_status == "On"):
                        equities[symbol]['status'] = "On" if new_status else "Off"
                        save_equities(equities)
                        st.rerun()
                
                # Display levels
                st.markdown("**Price Levels:**")
                levels_df = pd.DataFrame([
                    {"Level": k, "Price": f"${v:.2f}"} 
                    for k, v in data['levels'].items()
                ])
                st.dataframe(levels_df, width="stretch", hide_index=True)
                
                # Remove button
                if st.button(f"🗑️ Remove {symbol}", key=f"remove_{symbol}"):
                    del equities[symbol]
                    save_equities(equities)
                    st.success(f"Removed {symbol}")
                    time.sleep(1)
                    st.rerun()
    else:
        st.info("No trading systems configured. Add an equity in the 'Add Equity' tab.")

# Footer
st.markdown("---")
st.caption("⚠️ This is a paper trading bot. Always test thoroughly before using real money.")
