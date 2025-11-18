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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📚 Comment utiliser le Dashboard
    
    Le **Dashboard** est votre tableau de bord principal. C'est ici que vous visualisez l'état global de vos investissements.
    
    **Section "My Portfolio"** : Cette section affiche tous les actifs que vous possédez actuellement. Pour chaque position, 
    vous pouvez voir combien d'actions vous détenez, à quel prix moyen vous les avez achetées, leur prix actuel sur le marché, 
    et surtout votre gain ou perte (P&L). Le P&L vous indique si vous êtes en profit (positif) ou en perte (négatif) sur chaque investissement.
    
    **Section "Configured Trading Systems"** : Ici se trouvent les systèmes de trading automatisés que vous avez configurés. 
    Chaque système surveille un actif spécifique et peut exécuter des ordres automatiquement selon vos paramètres. 
    Vous pouvez rapidement supprimer un système avec les boutons 🗑️.
    
    ---
    
    #### 🎯 But de l'opération
    Le Dashboard est votre **centre de contrôle** pour surveiller la santé financière de votre portfolio. L'objectif est de vous donner 
    une vision instantanée de votre situation : Êtes-vous en profit global ? Quels actifs performent bien ou mal ? Combien valez-vous actuellement ?
    
    #### ⚖️ Les enjeux
    - **Visibilité totale** : Voir d'un coup d'œil si votre stratégie d'investissement fonctionne
    - **Détection rapide** : Identifier rapidement les positions problématiques (grosses pertes) ou gagnantes
    - **Contrôle des systèmes** : Surveiller quels systèmes automatisés sont actifs et peuvent trader pour vous
    - **Prise de décision** : Avoir toutes les informations nécessaires avant de modifier votre stratégie
    
    #### 💪 La difficulté
    La vraie difficulté n'est pas technique mais **psychologique** : il faut apprendre à regarder objectivement les chiffres, surtout les pertes, 
    sans panique ni euphorie. Un bon trader consulte son dashboard régulièrement mais ne prend pas de décisions impulsives. 
    Les marchés fluctuent constamment - ce qui compte c'est la tendance long terme, pas les variations quotidiennes.
    """)

# Tab 2: Buy/Sell
with tab2:
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📚 Comment utiliser Buy/Sell
    
    Cet onglet vous permet d'acheter et de vendre des actifs pour construire et gérer votre portfolio personnel.
    
    **📈 Buy Asset (Acheter)** : À gauche, vous pouvez acheter des actions. Entrez simplement le symbole boursier 
    (par exemple AAPL pour Apple, TSLA pour Tesla), choisissez la quantité, et confirmez le prix d'achat. 
    L'application calcule automatiquement le coût total de votre transaction. Si vous achetez plusieurs fois le même actif, 
    le système calcule automatiquement votre prix moyen d'achat.
    
    **📉 Sell Asset (Vendre)** : À droite, vous pouvez vendre les actifs que vous possédez déjà. Sélectionnez l'actif 
    dans la liste déroulante, choisissez combien d'actions vous voulez vendre, et le système calculera automatiquement 
    votre profit ou perte sur cette vente. C'est un excellent moyen de réaliser vos gains ou de limiter vos pertes.
    
    💡 **Astuce** : Gardez toujours un œil sur le P&L (Profit & Loss) avant de vendre pour prendre des décisions éclairées !
    
    ---
    
    #### 🎯 But de l'opération
    Vous êtes en train de **construire et gérer activement votre patrimoine financier**. Chaque achat est un pari sur l'avenir d'une entreprise 
    ou d'un actif. L'objectif est d'acheter à un prix que vous jugez intéressant et de vendre plus cher pour réaliser un profit. 
    C'est comme gérer une collection : vous achetez ce qui a de la valeur et vous vendez quand le prix est bon.
    
    #### ⚖️ Les enjeux
    - **Capital limité** : Chaque euro investi ici ne peut pas être investi ailleurs. Il faut choisir judicieusement.
    - **Timing** : Acheter trop cher ou vendre trop tôt peut transformer un bon investissement en perte
    - **Diversification** : Mettre tout son argent sur un seul actif est risqué. Il faut répartir intelligemment.
    - **Émotions** : L'avidité pousse à acheter quand tout monte (souvent trop cher), la peur pousse à vendre quand tout baisse (souvent trop tôt)
    
    #### 💪 La difficulté
    Le plus difficile est de **rester discipliné et rationnel**. Quand vous voyez un actif monter de 50%, l'envie de vendre pour sécuriser 
    le gain est forte. Mais parfois, il monte encore de 100% après ! À l'inverse, quand vous êtes à -20%, la panique vous pousse à vendre 
    pour limiter les dégâts, mais parfois c'est juste avant la remontée. La vraie difficulté est d'avoir un **plan clair avant d'acheter** : 
    "Je vends si ça monte à X% ou si ça baisse à Y%", et de s'y tenir quoi qu'il arrive. Sans plan, vous tradez avec vos émotions, 
    et les marchés punissent sévèrement les décisions émotionnelles.
    """)

# Tab 3: Add Equity
with tab3:
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📚 Comment utiliser Add Equity
    
    Cet onglet vous permet de **configurer des systèmes de trading automatisés** pour surveiller et trader des actifs spécifiques.
    
    **Configuration d'un système** : Entrez le symbole de l'actif que vous souhaitez trader (ex: SPY, MSFT, BTC), 
    puis définissez deux paramètres clés :
    
    - **Number of Levels (Nombre de niveaux)** : C'est le nombre de paliers de prix auxquels vous voulez acheter automatiquement. 
      Plus il y a de niveaux, plus votre système achètera à des prix différents.
    
    - **Drawdown % (Pourcentage de baisse)** : C'est l'écart de prix entre chaque niveau. Par exemple, avec un drawdown de 5%, 
      si le prix d'entrée est 100€, le niveau 1 sera à 95€, le niveau 2 à 90€, etc. Cela vous permet d'acheter progressivement 
      quand le prix baisse, réduisant ainsi votre prix moyen d'achat.
    
    ⚠️ **Limite** : Vous ne pouvez configurer que 10 systèmes maximum pour garder votre stratégie gérable et claire.
    
    💡 **Conseil** : Commencez avec un petit drawdown (2-3%) pour les actifs volatils, et un plus grand (5-10%) pour les actifs stables.
    
    ---
    
    #### 🎯 But de l'opération
    Vous êtes en train de mettre en place une **stratégie d'achat automatisée et disciplinée** appelée "Dollar Cost Averaging" (DCA) ou 
    "moyennage du prix d'achat". Au lieu d'investir tout votre capital d'un coup (risque de mal timer le marché), vous achetez petit à petit 
    à des prix décroissants. Si le marché baisse, vous accumulez plus d'actions à meilleur prix. Si ça remonte ensuite, vous êtes rentable 
    plus rapidement car votre prix moyen est bas. C'est une stratégie défensive qui transforme les baisses de marché en opportunités.
    
    #### ⚖️ Les enjeux
    - **Automatisation vs Contrôle** : Vous déléguez les décisions d'achat au système. Ça élimine les émotions mais nécessite une configuration réfléchie
    - **Capital requis** : Chaque niveau = un achat. 5 niveaux = potentiellement 5 achats. Assurez-vous d'avoir le capital nécessaire
    - **Choix du drawdown** : Trop petit (1-2%) = vous achetez souvent, peut-être trop tôt. Trop grand (10-15%) = vous ratez des opportunités
    - **Sélection des actifs** : Tous les actifs ne se valent pas. Un système DCA sur un actif en déclin permanent = pertes continues
    
    #### 💪 La difficulté
    La difficulté principale est de **calibrer correctement les paramètres** selon la volatilité de l'actif. Un actif stable comme SPY (S&P 500) 
    peut supporter un drawdown de 5-7% car il bouge lentement. Mais une crypto volatile comme BTC pourrait nécessiter 10-15% car elle peut 
    facilement faire -20% en une semaine. Si votre drawdown est trop petit, vous épuiserez tous vos niveaux rapidement sans avoir profité 
    des vraies occasions. Si c'est trop grand, les niveaux ne se déclencheront jamais et vous ne profiterez pas des petites baisses.
    
    Il faut aussi accepter psychologiquement de **voir son portfolio en rouge temporairement**. Un système DCA est conçu pour acheter en baisse, 
    donc par définition, après chaque achat, vous êtes souvent en perte papier. C'est normal et voulu ! La stratégie parie sur un retour 
    à la hausse à moyen terme. Si vous paniquez et désactivez tout au premier -10%, vous transformez des pertes temporaires en pertes réelles.
    """)

# Tab 4: AI Assistant
with tab4:
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📚 Comment utiliser l'AI Assistant
    
    Votre **assistant IA personnel** est un expert en analyse de portfolio et en marchés financiers. 
    Il a accès à toutes les données de votre portfolio et peut vous aider à prendre de meilleures décisions d'investissement.
    
    **Ce que vous pouvez lui demander** :
    - Analyser vos positions actuelles et identifier les risques
    - Évaluer la diversification de votre portfolio
    - Donner son avis sur un actif spécifique
    - Expliquer des concepts financiers complexes
    - Suggérer des ajustements à votre stratégie
    - Répondre à toutes vos questions sur le trading et l'investissement
    
    **Comment l'utiliser** : Posez simplement votre question dans le chat en bas de l'écran. L'IA analyse votre portfolio 
    en temps réel et vous fournit des réponses personnalisées basées sur vos positions actuelles.
    
    💡 **Exemples de questions** : "Mon portfolio est-il bien diversifié ?", "Quels sont les risques de mes positions actuelles ?", 
    "Devrais-je vendre AAPL maintenant ?", "Explique-moi ce qu'est le P&L".
    
    ⚠️ **Important** : L'IA donne des conseils éducatifs, mais la décision finale vous appartient toujours !
    
    ---
    
    #### 🎯 But de l'opération
    L'objectif est de vous donner un **second avis objectif et éduqué** sur vos décisions d'investissement. Quand on gère son propre argent, 
    on est souvent biaisé par nos émotions, nos espoirs, nos peurs. L'IA analyse froidement les données et vous donne une perspective 
    extérieure. C'est comme avoir un mentor financier disponible 24/7 qui connaît parfaitement votre situation.
    
    #### ⚖️ Les enjeux
    - **Éducation continue** : Chaque interaction vous aide à mieux comprendre les marchés et à devenir un investisseur plus avisé
    - **Détection de biais** : L'IA peut vous signaler si vous êtes trop exposé sur un secteur ou si vous prenez trop de risques
    - **Confirmation ou remise en question** : Parfois l'IA confirmera votre intuition, parfois elle vous alertera sur un danger que vous n'aviez pas vu
    - **Apprentissage des erreurs** : Demander pourquoi une position est en perte peut révéler des leçons importantes
    
    #### 💪 La difficulté
    La plus grande difficulté est de **savoir poser les bonnes questions**. L'IA est puissante mais elle répond à ce que vous demandez. 
    Si vous demandez "Dois-je acheter plus de TSLA ?", l'IA donnera un avis. Mais peut-être que la vraie question à poser était 
    "Mon portfolio est-il déjà surexposé aux actions tech ?" Une question plus large et stratégique.
    
    Autre piège : l'IA donne des analyses basées sur des données et des principes généraux, mais **elle ne prédit pas l'avenir**. 
    Elle peut dire "Historiquement, diversifier réduit le risque" (vrai), mais elle ne peut pas dire "AAPL va monter de 20% demain" (impossible à savoir).
    
    Enfin, il faut **rester critique**. L'IA est un outil d'aide, pas une boule de cristal. Si elle vous dit quelque chose qui semble 
    étrange ou contraire à votre compréhension, creusez davantage, posez des questions de suivi. L'objectif est de comprendre le "pourquoi" 
    derrière chaque conseil, pas juste d'obéir aveuglément.
    """)

# Tab 5: Trading Systems
with tab5:
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
    st.markdown("""
    ### 📚 Comment utiliser Trading Systems
    
    Cet onglet vous donne une **vue détaillée et un contrôle total** sur tous vos systèmes de trading automatisés.
    
    **Pour chaque système, vous pouvez** :
    - Voir tous les détails : position actuelle, prix d'entrée, drawdown configuré
    - **Activer/Désactiver** le système avec le bouton toggle. Quand "Active" est sur ON, le système surveille le marché 
      et peut passer des ordres automatiquement selon les niveaux de prix définis.
    - Visualiser tous les **niveaux de prix** : ce sont les prix auxquels le système achètera automatiquement. 
      Par exemple, si SPY baisse à 629.57€, le système achètera 1 action automatiquement (niveau 1), 
      puis une autre à 596.44€ (niveau 2), etc.
    - **Supprimer** un système si vous ne voulez plus l'utiliser
    
    **Comment ça fonctionne** : Imaginez que vous voulez acheter progressivement un actif quand son prix baisse. 
    Au lieu de surveiller le marché 24/7, le système le fait pour vous. Il achète automatiquement aux prix que vous avez définis, 
    réduisant ainsi votre prix moyen d'achat (stratégie DCA - Dollar Cost Averaging).
    
    💡 **Conseil** : Commencez par tester avec un seul système en mode "Off" pour comprendre comment fonctionnent les niveaux, 
    puis activez-le quand vous êtes à l'aise avec la stratégie.
    
    ⚠️ **Attention** : Un système actif peut passer des ordres automatiquement. Assurez-vous de comprendre votre stratégie avant d'activer !
    
    ---
    
    #### 🎯 But de l'opération
    Vous êtes en train de **piloter vos robots de trading**. C'est votre salle de contrôle. Chaque système que vous voyez ici est comme 
    un employé virtuel qui travaille pour vous, surveillant les marchés sans relâche et exécutant vos ordres selon votre stratégie. 
    L'objectif est d'avoir une vision claire de tous vos systèmes actifs, de pouvoir intervenir rapidement si nécessaire 
    (désactiver un système qui se comporte mal, ajuster des paramètres), et de monitorer l'efficacité de chaque stratégie.
    
    #### ⚖️ Les enjeux
    - **Cohérence stratégique** : Avoir 10 systèmes actifs signifie potentiellement 10 actifs différents. Sont-ils complémentaires ou redondants ?
    - **Gestion du capital** : Chaque système actif peut déclencher des achats. Avez-vous assez de capital pour tous les niveaux de tous les systèmes ?
    - **Surveillance continue** : Un système actif trade sans vous. Il faut vérifier régulièrement que les ordres exécutés sont cohérents
    - **Discipline système** : Résister à l'envie de désactiver un système au premier signe de baisse (qui est justement son moment d'action)
    
    #### 💪 La difficulté
    La difficulté majeure est de **faire confiance au système sans perdre le contrôle**. Quand vous activez un système, vous acceptez 
    qu'il achète automatiquement selon les niveaux configurés. Si le marché baisse fortement, le système va consommer vos niveaux un par un, 
    et votre compte sera en rouge. C'est exactement ce qui est prévu ! Mais humainement, c'est difficile à accepter. 
    
    Vous devez constamment résister à deux tentations opposées :
    1. **Sur-intervention** : Désactiver le système dès que ça baisse, modifier les niveaux constamment, micro-manager chaque trade. 
       Résultat : vous sabotez la stratégie et transformez un système discipliné en trading émotionnel.
    2. **Sous-surveillance** : Activer tous les systèmes et ne plus jamais regarder. Un système peut mal fonctionner (bug, mauvais paramétrage), 
       un actif peut s'effondrer durablement. Il faut un équilibre : faire confiance mais vérifier régulièrement.
    
    La clé est de **définir des règles claires avant d'activer** : "Je laisse ce système actif tant que l'actif ne baisse pas de plus de X%", 
    ou "Je révise mes systèmes tous les lundis". Avec des règles, vous évitez les décisions émotionnelles tout en gardant le contrôle.
    """)

# Footer
st.markdown("---")
st.caption("⚠️ This is a paper trading bot. Always test thoroughly before using real money.")
