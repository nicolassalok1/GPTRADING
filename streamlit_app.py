import streamlit as st
from openai import OpenAI
import os
import json
import alpaca_trade_api as tradeapi
import time
from dotenv import load_dotenv
import pandas as pd
import requests
import datetime

# Configuration
DATA_FILE = "equities.json"
PORTFOLIO_FILE = "portfolio.json"
SELL_SYSTEMS_FILE = "sell_systems.json"
OPTIONS_PORTFOLIO_FILE = "options_portfolio.json"
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


def load_sell_systems():
    try:
        with open(SELL_SYSTEMS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sell_systems(sell_systems):
    with open(SELL_SYSTEMS_FILE, 'w') as f:
        json.dump(sell_systems, f, indent=2)


def load_options_portfolio():
    try:
        with open(OPTIONS_PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_options_portfolio(options_portfolio):
    with open(OPTIONS_PORTFOLIO_FILE, 'w') as f:
        json.dump(options_portfolio, f, indent=2)

def buy_asset(symbol, quantity, price):
    portfolio = load_portfolio()
    now = time.strftime('%Y-%m-%d %H:%M:%S')

    position = portfolio.get(symbol)

    if position:
        old_qty = position['quantity']
        old_avg = position['avg_price']
        side = position.get('side', 'long')

        if side == 'long':
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
            portfolio[symbol] = {
                'quantity': new_qty,
                'avg_price': round(new_avg, 2),
                'side': 'long',
                'last_updated': now
            }
        else:
            if quantity < old_qty:
                new_qty = old_qty - quantity
                portfolio[symbol] = {
                    'quantity': new_qty,
                    'avg_price': old_avg,
                    'side': 'short',
                    'last_updated': now
                }
            elif quantity == old_qty:
                del portfolio[symbol]
            else:
                new_long_qty = quantity - old_qty
                portfolio[symbol] = {
                    'quantity': new_long_qty,
                    'avg_price': round(price, 2),
                    'side': 'long',
                    'last_updated': now
                }
    else:
        portfolio[symbol] = {
            'quantity': quantity,
            'avg_price': round(price, 2),
            'side': 'long',
            'last_updated': now
        }
    save_portfolio(portfolio)
    return portfolio.get(symbol)


def sell_asset(symbol, quantity, price):
    portfolio = load_portfolio()
    now = time.strftime('%Y-%m-%d %H:%M:%S')

    position = portfolio.get(symbol)

    if position:
        old_qty = position['quantity']
        old_avg = position['avg_price']
        side = position.get('side', 'long')

        if side == 'long':
            if quantity < old_qty:
                new_qty = old_qty - quantity
                portfolio[symbol] = {
                    'quantity': new_qty,
                    'avg_price': old_avg,
                    'side': 'long',
                    'last_updated': now
                }
            elif quantity == old_qty:
                del portfolio[symbol]
            else:
                new_short_qty = quantity - old_qty
                portfolio[symbol] = {
                    'quantity': new_short_qty,
                    'avg_price': round(price, 2),
                    'side': 'short',
                    'last_updated': now
                }
        else:
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
            portfolio[symbol] = {
                'quantity': new_qty,
                'avg_price': round(new_avg, 2),
                'side': 'short',
                'last_updated': now
            }
    else:
        portfolio[symbol] = {
            'quantity': quantity,
            'avg_price': round(price, 2),
            'side': 'short',
            'last_updated': now
        }

    save_portfolio(portfolio)
    return True


def process_sell_systems():
    sell_systems = load_sell_systems()
    if not sell_systems:
        st.info("No sell systems configured.")
        return

    portfolio = load_portfolio()
    any_executed = False

    for symbol, config in sell_systems.items():
        if config.get("status") != "On":
            continue
        if symbol not in portfolio:
            continue

        current_price_data = get_data(symbol)
        current_price = current_price_data['price']
        if current_price <= 0:
            continue

        levels = config.get("levels", {})
        for level_key, level in levels.items():
            if level.get("triggered"):
                continue

            trigger_price = level.get("price")
            level_qty = int(level.get("quantity", 0))
            if trigger_price is None or level_qty <= 0:
                continue

            current_position_qty = portfolio.get(symbol, {}).get("quantity", 0)
            if current_position_qty <= 0:
                break

            if current_price <= trigger_price:
                qty_to_sell = min(level_qty, current_position_qty)
                if qty_to_sell <= 0:
                    continue

                if sell_asset(symbol, qty_to_sell, current_price):
                    level["triggered"] = True
                    any_executed = True
                    st.success(
                        f"Auto-sell executed for {symbol}: sold {qty_to_sell} units "
                        f"around ${current_price:.2f} (level {level_key})"
                    )
                    portfolio = load_portfolio()

        config["levels"] = levels

    if any_executed:
        save_sell_systems(sell_systems)
    else:
        st.info("No sell levels were triggered based on current market prices.")


def trade_option_contract(
    contract_symbol,
    underlying_symbol,
    option_type,
    strike,
    expiration,
    side,
    quantity,
    price,
):
    options_portfolio = load_options_portfolio()
    now = time.strftime('%Y-%m-%d %H:%M:%S')

    position = options_portfolio.get(contract_symbol)

    if position:
        old_qty = position['quantity']
        old_avg = position['avg_price']
        old_side = position.get('side', 'long')

        if old_side == side:
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
            options_portfolio[contract_symbol] = {
                'underlying': underlying_symbol,
                'type': option_type,
                'strike': strike,
                'expiration': expiration,
                'quantity': new_qty,
                'avg_price': round(new_avg, 4),
                'side': side,
                'last_updated': now,
            }
        else:
            if quantity < old_qty:
                new_qty = old_qty - quantity
                options_portfolio[contract_symbol] = {
                    'underlying': underlying_symbol,
                    'type': option_type,
                    'strike': strike,
                    'expiration': expiration,
                    'quantity': new_qty,
                    'avg_price': old_avg,
                    'side': old_side,
                    'last_updated': now,
                }
            elif quantity == old_qty:
                del options_portfolio[contract_symbol]
            else:
                new_qty = quantity - old_qty
                options_portfolio[contract_symbol] = {
                    'underlying': underlying_symbol,
                    'type': option_type,
                    'strike': strike,
                    'expiration': expiration,
                    'quantity': new_qty,
                    'avg_price': round(price, 4),
                    'side': side,
                    'last_updated': now,
                }
    else:
        options_portfolio[contract_symbol] = {
            'underlying': underlying_symbol,
            'type': option_type,
            'strike': strike,
            'expiration': expiration,
            'quantity': quantity,
            'avg_price': round(price, 4),
            'side': side,
            'last_updated': now,
        }

    save_options_portfolio(options_portfolio)
    return options_portfolio.get(contract_symbol)


def fetch_options_chain(symbol):
    try:
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol.upper()}.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        payload = response.json()
        data_block = payload.get("data", {})
        options = data_block.get("options", [])
        spot = data_block.get("current_price", 0.0) or 0.0

        today = datetime.date.today()
        chain = []

        for c in options:
            opt_symbol = c.get("option")
            if not opt_symbol or len(opt_symbol) < 15:
                continue

            # OCC symbology: root + YYMMDD + C/P + 8-digit strike
            date_code = opt_symbol[-15:-9]  # YYMMDD
            cp_flag = opt_symbol[-9]        # C or P
            strike_code = opt_symbol[-8:]   # 8 digits

            try:
                year = 2000 + int(date_code[0:2])
                month = int(date_code[2:4])
                day = int(date_code[4:6])
                expiration = datetime.date(year, month, day)
            except Exception:
                continue

            try:
                strike = int(strike_code) / 1000.0
            except Exception:
                continue

            days_to_expiry = (expiration - today).days
            T = max(days_to_expiry, 0) / 365.0

            bid = c.get("bid", 0.0) or 0.0
            ask = c.get("ask", 0.0) or 0.0
            last = c.get("last_trade_price", 0.0) or 0.0

            if bid > 0 and ask > 0:
                price = (bid + ask) / 2.0
            elif last > 0:
                price = last
            else:
                price = max(bid, ask)

            iv = c.get("iv", 0.0) or 0.0

            chain.append(
                {
                    "symbol": opt_symbol,
                    "underlying": data_block.get("symbol", symbol.upper()),
                    "spot": spot,
                    "type": "call" if cp_flag.upper() == "C" else "put",
                    "strike": strike,
                    "expiration": expiration.isoformat(),
                    "T": T,
                    "price": price,
                    "iv": iv,
                }
            )

        return chain
    except Exception as e:
        st.error(f"Error fetching options from CBOE for {symbol}: {e}")
        return []

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
st.title("📈 AI assisted Trading system")
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "💰 Buy/Sell",
    "➕ Add Equity",
    "📋 Trading Systems",
    "🧾 Options",
])

# Tab 1: Dashboard
with tab1:
    # AI Assistant section moved here
    st.subheader("🤖 AI Portfolio Assistant")
    st.markdown("Ask questions about your portfolio and get AI-powered insights")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about your portfolio..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatgpt_response(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown("---")

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
        total_pnl = 0
        total_notional = 0

        for symbol, data in my_portfolio.items():
            current_price_data = get_data(symbol)
            avg_price = data['avg_price']
            quantity = data['quantity']
            side = data.get('side', 'long')

            current_price = current_price_data['price'] if current_price_data['price'] > 0 else avg_price
            position_value = quantity * current_price

            if side == 'long':
                pnl = (current_price - avg_price) * quantity
            else:
                pnl = (avg_price - current_price) * quantity

            notional = avg_price * quantity
            pnl_pct = (pnl / notional * 100) if notional > 0 else 0

            total_value += position_value
            total_pnl += pnl
            total_notional += notional
            
            portfolio_data.append({
                'Symbol': symbol,
                'Side': side.capitalize(),
                'Quantity': quantity,
                'Avg Price': f"${avg_price:.2f}",
                'Current Price': f"${current_price:.2f}",
                'Value': f"${position_value:.2f}",
                'P&L': f"${pnl:.2f}",
                'P&L %': f"{pnl_pct:.2f}%"
            })
        
        df_my_portfolio = pd.DataFrame(portfolio_data)
        st.dataframe(df_my_portfolio, width="stretch", hide_index=True)

        total_pnl_pct = (total_pnl / total_notional * 100) if total_notional > 0 else 0
        val_col, pnl_col = st.columns(2)
        with val_col:
            st.metric("Total Portfolio Value", f"${total_value:.2f}")
        with pnl_col:
            st.metric("Total P&L", f"${total_pnl:.2f}", delta=f"{total_pnl_pct:.2f}%")

        # Options portfolio section
        st.markdown("---")
        st.markdown("### 📂 Options Portfolio")
        options_portfolio = load_options_portfolio()
        if options_portfolio:
            options_rows = []
            for contract_symbol, pos in options_portfolio.items():
                options_rows.append({
                    "Contract": contract_symbol,
                    "Underlying": pos.get("underlying"),
                    "Type": pos.get("type", "").capitalize(),
                    "Side": pos.get("side", "long").capitalize(),
                    "Strike": pos.get("strike"),
                    "Expiration": pos.get("expiration"),
                    "Quantity": pos.get("quantity"),
                    "Avg Price": f"${pos.get('avg_price', 0):.4f}",
                })
            df_options = pd.DataFrame(options_rows)
            st.dataframe(df_options, width="stretch", hide_index=True)
        else:
            st.info("No options positions in portfolio yet.")
    else:
        st.info("No assets in portfolio. Use the Buy/Sell tab to add positions.")
    
    # Trading Systems Section
    st.markdown("---")
    st.subheader("🎯 Configured Trading Systems")
    equities = load_equities()
    
    if equities:
        systems_data = []
        for symbol, data in equities.items():
            direction = data.get('direction', 'long')
            systems_data.append({
                'Symbol': symbol,
                'Direction': direction.capitalize(),
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
        st.markdown("### 📈 Buy / Cover Asset")
        buy_side = st.radio(
            "Direction",
            options=["Long", "Short"],
            index=0,
            horizontal=True,
            key="buy_side"
        )
        buy_symbol = st.text_input("Symbol to Buy", placeholder="e.g., AAPL", key="buy_symbol").upper()
        buy_quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="buy_qty")
        
        if buy_symbol:
            price_data = get_data(buy_symbol)
            if price_data['price'] > 0:
                st.info(f"Current price: ${price_data['price']:.2f}")
                buy_price = st.number_input("Buy Price", min_value=0.01, value=float(price_data['price']), step=0.01, key="buy_price")
                total_cost = buy_quantity * buy_price
                st.metric("Total Cost", f"${total_cost:.2f}")
                
                if st.button("✅ Execute Order", type="primary", key="exec_buy"):
                    if buy_side == "Long":
                        result = buy_asset(buy_symbol, buy_quantity, buy_price)
                        st.success(f"Bought {buy_quantity} units of {buy_symbol} @ ${buy_price:.2f}")
                        if result:
                            side = result.get('side', 'long').upper()
                            st.info(
                                f"New position: {result['quantity']} units @ avg ${result['avg_price']:.2f} "
                                f"({side})"
                            )
                        else:
                            st.info("Position fully closed.")
                    else:
                        if sell_asset(buy_symbol, buy_quantity, buy_price):
                            st.success(f"Shorted {buy_quantity} units of {buy_symbol} @ ${buy_price:.2f}")
                            portfolio_after = load_portfolio()
                            new_pos = portfolio_after.get(buy_symbol)
                            if new_pos:
                                side = new_pos.get("side", "short").upper()
                                st.info(
                                    f"New position: {new_pos['quantity']} units @ avg ${new_pos['avg_price']:.2f} "
                                    f"({side})"
                                )
                    time.sleep(1)
                    st.rerun()
            else:
                st.error(f"Could not fetch price for {buy_symbol}")
    
    # SELL / SHORT Section
    with col2:
        st.markdown("### 📉 Sell / Short Asset")
        my_portfolio = load_portfolio()
        
        if my_portfolio:
            sell_symbol = st.selectbox("Symbol to Sell/Short", options=list(my_portfolio.keys()), key="sell_symbol")
            
            if sell_symbol:
                position = my_portfolio[sell_symbol]
                current_qty = position['quantity']
                avg_price = position['avg_price']
                side = position.get('side', 'long')
                
                st.info(
                    f"Current position: {current_qty} units @ avg ${avg_price:.2f} "
                    f"({side.upper()})"
                )
                
                sell_quantity = st.number_input(
                    "Quantity to Sell (you can sell more than you hold to go net short)",
                    min_value=1,
                    value=1,
                    step=1,
                    key="sell_qty"
                )
                
                price_data = get_data(sell_symbol)
                if price_data['price'] > 0:
                    sell_price = st.number_input(
                        "Sell Price",
                        min_value=0.01,
                        value=float(price_data['price']),
                        step=0.01,
                        key="sell_price"
                    )
                    total_proceeds = sell_quantity * sell_price
                    if side == 'long':
                        pnl = (sell_price - avg_price) * sell_quantity
                    else:
                        pnl = (avg_price - sell_price) * sell_quantity
                    notional = avg_price * sell_quantity
                    pnl_pct = (pnl / notional * 100) if notional > 0 else 0.0
                    
                    st.metric("Total Proceeds", f"${total_proceeds:.2f}")
                    st.metric("P&L (per this trade)", f"${pnl:.2f}", delta=f"{pnl_pct:.2f}%")
                    
                    if st.button("✅ Execute Sell / Short", type="primary", key="exec_sell"):
                        if sell_asset(sell_symbol, sell_quantity, sell_price):
                            st.success(f"Sold {sell_quantity} units of {sell_symbol} @ ${sell_price:.2f}")
                            st.info(f"Trade P&L (approx.): ${pnl:.2f}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to execute sell order")
                else:
                    st.error(f"Could not fetch price for {sell_symbol}")
        else:
            st.info("No assets in portfolio to sell or short")

# Tab 3: Add Equity
with tab3:
    with st.expander("📘 Comprendre Add Equity"):
        st.markdown("""
        ### ➕ Ce que vous faites dans Add Equity
        
        Ici, vous ne passez pas d'ordres immédiats : vous **concevez des systèmes automatiques**
        qui vont acheter ou vendre à découvert pour vous selon une logique de drawdown et de niveaux de prix.
        
        Le champ *Symbol* sert à choisir l'actif que vous voulez suivre
        de façon structurée (indice, action, ETF, crypto, etc.).
        
        Le champ *Direction* vous permet de choisir si vous pariez sur la hausse (Long) ou la baisse (Short) :
        - **Long** : position acheteuse sur l'actif
        - **Short** : position vendeuse à découvert sur l'actif
        
        *Number of Levels* définit combien de paliers d'achat/vente vous voulez.
        Chaque niveau correspondra à un prix où le système interviendra automatiquement.
        
        *Drawdown %* contrôle l'écart et la direction des niveaux de prix :
        - **Valeur négative** (ex: -5%) : les niveaux seront **à la baisse** (en dessous du prix d'entrée)
          → Utile pour acheter plus bas (Long) ou couvrir des shorts en baisse
        - **Valeur positive** (ex: +5%) : les niveaux seront **à la hausse** (au dessus du prix d'entrée)
          → Utile pour shorter plus haut (Short) ou pyramider sur une hausse (Long)
        
        Un pourcentage faible donnera des niveaux proches les uns des autres,
        un pourcentage plus élevé espacera davantage les interventions.
        
        Quand vous cliquez sur *Add Equity*, l'outil calcule les niveaux
        à partir du prix actuel et enregistre un système en mode *Off* par défaut,
        que vous pourrez ensuite activer et superviser dans l'onglet *Trading Systems*.
        
        Utilisez cet onglet pour **planifier à l'avance** comment vous voulez intervenir
        pendant les mouvements de marché, plutôt que d'improviser dans la panique.
        """)

    st.subheader("➕ Add New Equity")
    
    # Check current count
    equities = load_equities()
    current_count = len(equities)
    
    st.info(f"Trading Systems: {current_count}/10")
    
    if current_count >= 10:
        st.error("⚠️ Maximum limit reached! You cannot add more than 10 equities. Please remove one first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbol = st.text_input("Symbol", placeholder="e.g., AAPL").upper()
        
        with col2:
            direction = st.radio("Direction", options=["Long", "Short"], index=0, horizontal=True, key="add_equity_direction")
        
        with col3:
            levels = st.number_input("Number of Levels", min_value=1, max_value=10, value=5)
        
        with col4:
            drawdown = st.number_input("Drawdown %", min_value=-50.0, max_value=50.0, value=5.0, step=0.1)
        
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
                        
                        # Si drawdown négatif: niveaux à la baisse (en dessous du prix d'entrée)
                        # Si drawdown positif: niveaux à la hausse (au dessus du prix d'entrée)
                        if drawdown_decimal < 0:
                            # Drawdown négatif → niveaux en dessous
                            level_prices = {str(i+1): round(entry_price * (1 + drawdown_decimal * (i+1)), 2) for i in range(levels)}
                        else:
                            # Drawdown positif → niveaux au dessus
                            level_prices = {str(i+1): round(entry_price * (1 + drawdown_decimal * (i+1)), 2) for i in range(levels)}
                        
                        stored_drawdown = drawdown_decimal
                        
                        equities[symbol] = {
                            "position": 0,
                            "entry_price": entry_price,
                            "levels": level_prices,
                            "drawdown": stored_drawdown,
                            "direction": direction.lower(),
                            "status": "Off"
                        }
                        
                        save_equities(equities)
                        st.success(f"✅ Added {symbol} ({direction}) at ${entry_price:.2f}")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Could not fetch price for {symbol}")
            else:
                st.error("Please enter a symbol")

# Tab 4: Trading Systems
with tab4:
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
            direction = data.get('direction', 'long')
            with st.expander(f"{symbol} ({direction.upper()}) - Status: {data['status']}", expanded=True):
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Direction", direction.capitalize())
                
                with col2:
                    st.metric("Position", data['position'])
                
                with col3:
                    st.metric("Entry Price", f"${data['entry_price']:.2f}")
                
                with col4:
                    st.metric("Drawdown", f"{data['drawdown']*100:.1f}%")
                
                with col5:
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

# Tab 5: Options
with tab5:
    with st.expander("📘 Comprendre Options"):
        st.markdown("""
        ### 🧾 Ce que vous faites dans Options
        
        Cet onglet vous permet d'explorer les **options européennes** liées à un sous-jacent,
        puis de prendre des positions longues (achat) ou courtes (vente à découvert) sur un contrat précis.
        
        L'idée est de vous donner un outil complémentaire aux actions :
        vous pouvez exprimer une vue directionnelle avec effet de levier,
        couvrir une position existante, ou structurer des paris plus finement
        qu'avec le sous-jacent seul.
        
        Le flux de travail est le suivant :
        1. Vous choisissez un **ticker sous-jacent** (par exemple AAPL)
        2. L'application récupère une liste de contrats d'options européennes disponibles
        3. Vous sélectionnez un contrat (type, strike, échéance)
        4. Vous choisissez si vous voulez être **Long** (acheteur de l'option) ou **Short** (vendeur)
        5. Vous entrez la quantité et le prix, puis vous validez l'ordre
        
        Toutes vos positions options sont ensuite visibles dans la section *Options Portfolio*
        du Dashboard, avec le sens (Long/Short) clairement indiqué.
        """)

    st.subheader("🧾 Trade Options (European)")

    underlying_symbol = st.text_input(
        "Underlying symbol for options",
        placeholder="e.g., AAPL",
        key="opt_underlying",
    ).upper()

    if "options_chain" not in st.session_state:
        st.session_state.options_chain = []

    if underlying_symbol:
        if st.button("🔍 Load European options chain", key="load_options_chain"):
            st.session_state.options_chain = fetch_options_chain(underlying_symbol)

    options_chain = st.session_state.options_chain

    # Filter chain to current underlying only
    filtered_chain = [
        c for c in options_chain
        if str(c.get("underlying", "")).upper() == underlying_symbol.upper()
    ] if underlying_symbol and options_chain else []

    if underlying_symbol and filtered_chain:
        # Step 1: choose maturity T (years) with a selectbox (unique T values, rounded to 2 decimals)
        unique_T_values = sorted({round(float(c["T"]), 2) for c in filtered_chain})

        if unique_T_values:
            display_T_values = [f"{t:.2f}" for t in unique_T_values]
            selected_T_display = st.selectbox(
                "Select maturity T (years)",
                options=display_T_values,
                key="opt_selected_T",
            )
            try:
                selected_T = float(selected_T_display)
            except ValueError:
                selected_T = None
        else:
            selected_T = None

        if selected_T is not None:
            filtered_by_T = [
                c for c in filtered_chain if round(c["T"], 2) == round(selected_T, 2)
            ]

            if filtered_by_T:
                st.markdown("#### Select strike (K) for chosen T")

                tab_call, tab_put = st.tabs(["Call", "Put"])

                # Calls tab
                with tab_call:
                    calls_for_T = [c for c in filtered_by_T if c["type"] == "call"]
                    if not calls_for_T:
                        st.info("No call options for this maturity.")
                    else:
                        unique_K_call = sorted({float(c["strike"]) for c in calls_for_T})
                        if unique_K_call:
                            # Slider directly on K values (label shows K)
                            min_k_call = float(min(unique_K_call))
                            max_k_call = float(max(unique_K_call))
                            if len(unique_K_call) > 1:
                                diffs_call = [
                                    b - a for a, b in zip(unique_K_call[:-1], unique_K_call[1:])
                                ]
                                step_call = min(diffs_call)
                            else:
                                step_call = 1.0

                            default_k_call = unique_K_call[min(
                                len(unique_K_call) // 2, len(unique_K_call) - 1
                            )]

                            selected_k_value_call = st.slider(
                                "Strike (K) - Call",
                                min_value=min_k_call,
                                max_value=max_k_call,
                                value=float(default_k_call),
                                step=float(step_call),
                                format="%.2f",
                                key="opt_call_strike",
                            )

                            # Map chosen K to nearest available strike
                            selected_K_call = min(
                                unique_K_call,
                                key=lambda k: abs(k - selected_k_value_call),
                            )
                            selected_call = next(
                                (c for c in calls_for_T if c["strike"] == selected_K_call),
                                None,
                            )

                            if selected_call:
                                spot_call = float(selected_call.get("spot", 0.0) or 0.0)
                                chips_html_call = f"""
                                <div style='display:flex;flex-wrap:wrap;gap:0.6rem;margin-top:0.6rem;'>
                                  <span style='background-color:#e3f2fd;color:#0d47a1;padding:6px 14px;border-radius:999px;font-size:0.9rem;font-weight:600;'>Call</span>
                                  <span style='background-color:#fffde7;color:#f57f17;padding:6px 14px;border-radius:999px;font-size:0.9rem;'>Spot = {spot_call:.2f}</span>
                                  <span style='background-color:#e8f5e9;color:#1b5e20;padding:6px 14px;border-radius:999px;font-size:0.9rem;'>T = {selected_call['T']:.2f}</span>
                                  <span style='background-color:#fff3e0;color:#e65100;padding:6px 14px;border-radius:999px;font-size:0.9rem;'>K = {selected_call['strike']:.2f}</span>
                                  <span style='background-color:#f3e5f5;color:#4a148c;padding:6px 14px;border-radius:999px;font-size:0.9rem;'>Price = {selected_call['price']:.2f}</span>
                                  <span style='background-color:#fce4ec;color:#880e4f;padding:6px 14px;border-radius:999px;font-size:0.9rem;'>IV = {selected_call['iv']:.2f}</span>
                                </div>
                                """
                                st.markdown(chips_html_call, unsafe_allow_html=True)

                                side_call = st.radio(
                                    "Position side",
                                    options=["Long", "Short"],
                                    horizontal=True,
                                    key="opt_side_call",
                                )
                                qty_call = st.number_input(
                                    "Quantity (contracts)",
                                    min_value=1,
                                    value=1,
                                    step=1,
                                    key="opt_qty_call",
                                )
                                default_price_call = (
                                    float(f"{selected_call['price']:.2f}")
                                    if selected_call["price"] > 0
                                    else 1.00
                                )
                                price_call = st.number_input(
                                    "Price per contract",
                                    min_value=0.01,
                                    value=default_price_call,
                                    step=0.01,
                                    key="opt_price_call",
                                )

                                if st.button(
                                    "✅ Execute Call Trade",
                                    type="primary",
                                    key="exec_option_call",
                                ):
                                    result = trade_option_contract(
                                        contract_symbol=selected_call["symbol"],
                                        underlying_symbol=selected_call.get("underlying"),
                                        option_type=selected_call.get("type"),
                                        strike=selected_call.get("strike"),
                                        expiration=selected_call.get("expiration"),
                                        side=side_call.lower(),
                                        quantity=int(qty_call),
                                        price=float(price_call),
                                    )
                                    if result:
                                        st.success(
                                            f"{side_call} {qty_call}x {selected_call['symbol']} "
                                            f"@ {price_call:.2f} recorded in options portfolio."
                                        )
                                    else:
                                        st.success("Option position updated / closed.")

                # Puts tab
                with tab_put:
                    puts_for_T = [c for c in filtered_by_T if c["type"] == "put"]
                    if not puts_for_T:
                        st.info("No put options for this maturity.")
                    else:
                        unique_K_put = sorted({float(c["strike"]) for c in puts_for_T})
                        if unique_K_put:
                            min_k_put = float(min(unique_K_put))
                            max_k_put = float(max(unique_K_put))
                            if len(unique_K_put) > 1:
                                diffs_put = [
                                    b - a for a, b in zip(unique_K_put[:-1], unique_K_put[1:])
                                ]
                                step_put = min(diffs_put)
                            else:
                                step_put = 1.0

                            default_k_put = unique_K_put[min(
                                len(unique_K_put) // 2, len(unique_K_put) - 1
                            )]

                            selected_k_value_put = st.slider(
                                "Strike (K) - Put",
                                min_value=min_k_put,
                                max_value=max_k_put,
                                value=float(default_k_put),
                                step=float(step_put),
                                format="%.2f",
                                key="opt_put_strike",
                            )

                            selected_K_put = min(
                                unique_K_put,
                                key=lambda k: abs(k - selected_k_value_put),
                            )
                            selected_put = next(
                                (c for c in puts_for_T if c["strike"] == selected_K_put),
                                None,
                            )

                            if selected_put:
                                spot_put = float(selected_put.get("spot", 0.0) or 0.0)
                                chips_html_put = f"""
                                <div style='display:flex;flex-wrap:wrap;gap:0.6rem;margin-top:0.6rem;'>
                                  <span style='background-color:#ffebee;color:#b71c1c;padding:6px 14px;border-radius:999px;font-size:0.9rem;font-weight:600;'>Put</span>
                                  <span style='background-color:#fffde7;color:#f57f17;padding:6px 14px;border-radius:999px;font-size:0.9rem;'>Spot = {spot_put:.2f}</span>
                                  <span style='background-color:#e8f5e9;color:#1b5e20;padding:6px 14px;border-radius:999px;font-size:0.9rem;'>T = {selected_put['T']:.2f}</span>
                                  <span style='background-color:#fff3e0;color:#e65100;padding:6px 14px;border-radius:999px;font-size:0.9rem;'>K = {selected_put['strike']:.2f}</span>
                                  <span style='background-color:#ede7f6;color:#311b92;padding:6px 14px;border-radius:999px;font-size:0.9rem;'>Price = {selected_put['price']:.2f}</span>
                                  <span style='background-color:#e0f7fa;color:#006064;padding:6px 14px;border-radius:999px;font-size:0.9rem;'>IV = {selected_put['iv']:.2f}</span>
                                </div>
                                """
                                st.markdown(chips_html_put, unsafe_allow_html=True)

                                side_put = st.radio(
                                    "Position side",
                                    options=["Long", "Short"],
                                    horizontal=True,
                                    key="opt_side_put",
                                )
                                qty_put = st.number_input(
                                    "Quantity (contracts)",
                                    min_value=1,
                                    value=1,
                                    step=1,
                                    key="opt_qty_put",
                                )
                                default_price_put = (
                                    float(f"{selected_put['price']:.2f}")
                                    if selected_put["price"] > 0
                                    else 1.00
                                )
                                price_put = st.number_input(
                                    "Price per contract",
                                    min_value=0.01,
                                    value=default_price_put,
                                    step=0.01,
                                    key="opt_price_put",
                                )

                                if st.button(
                                    "✅ Execute Put Trade",
                                    type="primary",
                                    key="exec_option_put",
                                ):
                                    result = trade_option_contract(
                                        contract_symbol=selected_put["symbol"],
                                        underlying_symbol=selected_put.get("underlying"),
                                        option_type=selected_put.get("type"),
                                        strike=selected_put.get("strike"),
                                        expiration=selected_put.get("expiration"),
                                        side=side_put.lower(),
                                        quantity=int(qty_put),
                                        price=float(price_put),
                                    )
                                    if result:
                                        st.success(
                                            f"{side_put} {qty_put}x {selected_put['symbol']} "
                                            f"@ {price_put:.2f} recorded in options portfolio."
                                        )
                                    else:
                                        st.success("Option position updated / closed.")

    elif underlying_symbol and not filtered_chain:
        st.info("No European options found or failed to load chain for this symbol.")

# Footer
st.markdown("---")
st.caption("⚠️ This is a paper trading bot. Always test thoroughly before using real money.")
