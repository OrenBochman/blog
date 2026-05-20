## Goal:

Create Simulated data sets

## steps:

1.  break down simulation into blocks
2.  simulate each block into a csv
3.  simulate an event stream
4.  simulate a graph
5.  put on s3
6.  put in deltalake
7.  process with spark

# Block list

Simulated Fake data is one of the most important tools in the Bayesian arsnal.

1.  it makes us think about the data side of the problem.
2.  it lets find out about some of the contraints that should go into building the priors.
3.  it gives us a chorent ground truth to validate our models.

In this case I want to simulate an ecommerce store: the first two objects have to do with demand

1.  there are different types of seasonality \[winter,summer,spring,fall,weekdays,weekend,holiday,none\]
2.  there are different types of trend \[up,down,steady,random\]
3.  there are thee relations between products \[substitutes, complements, none\]
4.  there is three relation beteen products and brands \[inferior, superior, none\]
5.  there is a list of brands
6.  there is a list of product categories.
7.  there is a list of products.

generate_brand() id, rv_quality: Normal(`quality_mean`, `quality_variance`) name product_generator(brand,category): generate a product with fields: id, brand, description, quality_score drawn from the brand’s distribution for each product category: while there are less than three products in the category() for each brand: if there is a product with the brand continue else if random bernulli(p=0.5) \> 0.3333 generate a product .

prompt

``` {text}
can you write a python generator that takes parmeters `user_count` for the number of user `start_date` and simulates user events in an ecommerce site. The events are drawn from a markov cain with the following states [entry, product_view, add_to_cart, check_in, sale] each event has a time_stamp, user_id, session_id, and product. The event is one of  if the page is a sale there is a purchase price .
```

``` python
import random
import datetime
STATES = ['entry', 'product_view', 'add_to_cart', 'check_in', 'sale']

# Define the transition probabilities between states
TRANSITION_PROBABILITIES = {
    'entry': {'product_view': 0.5, 'check_in': 0.3, 'add_to_cart': 0.2},
    'product_view': {'add_to_cart': 0.3, 'check_in': 0.5, 'entry': 0.2},
    'add_to_cart': {'check_in': 0.4, 'product_view': 0.3, 'sale': 0.3},
    'check_in': {'product_view': 0.4, 'add_to_cart': 0.3, 'sale': 0.3},
    'sale': {}
}

def generate_events(user_count, start_date):
    for user_id in range(1, user_count + 1):
        # Generate a session ID for the user
        session_id = str(user_id) + '_' + str(random.randint(100000, 999999))
        # Initialize the state and timestamp for the user's first event
        current_state = 'entry'
        current_time = start_date
        while True:
            # Generate the next state based on the current state and transition probabilities
            next_state = random.choices(list(TRANSITION_PROBABILITIES[current_state].keys()), 
                                         list(TRANSITION_PROBABILITIES[current_state].values()))[0]
            # Generate a timestamp for the next event
            time_diff = datetime.timedelta(seconds=random.randint(1, 300))
            next_time = current_time + time_diff
            # Generate a product ID for product-related events
            product = None
            if next_state in ['product_view', 'add_to_cart', 'sale']:
                product = 'product_' + str(random.randint(1, 100))
            # If the next state is a sale, generate a purchase price
            purchase_price = None
            if next_state == 'sale':
                purchase_price = round(random.uniform(10, 100), 2)
            # Yield the event with the user ID, session ID, state, timestamp, product, and purchase price
            yield {'user_id': user_id, 'session_id': session_id, 'state': next_state, 'time_stamp': next_time,
                   'product': product, 'purchase_price': purchase_price}
            # If the next state is a sale, stop generating events for the user
            if next_state == 'sale':
                break
            # Otherwise, update the current state and timestamp for the next event
            current_state = next_state
            current_time = next_time
```

``` python
# Generate events for 10 users starting from January 1, 2023
events = generate_events(user_count=10, start_date=datetime.datetime(2023, 1, 1))

# Iterate through the events and print them
for event in events:
    print(event)
```

    {'user_id': 1, 'session_id': '1_832890', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 3, 14), 'product': None, 'purchase_price': None}
    {'user_id': 1, 'session_id': '1_832890', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 4, 19), 'product': 'product_37', 'purchase_price': None}
    {'user_id': 1, 'session_id': '1_832890', 'state': 'entry', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 9, 16), 'product': None, 'purchase_price': None}
    {'user_id': 1, 'session_id': '1_832890', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 11, 21), 'product': None, 'purchase_price': None}
    {'user_id': 1, 'session_id': '1_832890', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 12, 23), 'product': 'product_93', 'purchase_price': None}
    {'user_id': 1, 'session_id': '1_832890', 'state': 'sale', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 15, 14), 'product': 'product_90', 'purchase_price': 82.94}
    {'user_id': 2, 'session_id': '2_903009', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 4, 35), 'product': 'product_82', 'purchase_price': None}
    {'user_id': 2, 'session_id': '2_903009', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 5, 35), 'product': None, 'purchase_price': None}
    {'user_id': 2, 'session_id': '2_903009', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 6, 4), 'product': 'product_17', 'purchase_price': None}
    {'user_id': 2, 'session_id': '2_903009', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 7, 45), 'product': 'product_61', 'purchase_price': None}
    {'user_id': 2, 'session_id': '2_903009', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 9, 41), 'product': None, 'purchase_price': None}
    {'user_id': 2, 'session_id': '2_903009', 'state': 'sale', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 14, 16), 'product': 'product_4', 'purchase_price': 42.23}
    {'user_id': 3, 'session_id': '3_668821', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 4, 17), 'product': 'product_8', 'purchase_price': None}
    {'user_id': 3, 'session_id': '3_668821', 'state': 'sale', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 4, 54), 'product': 'product_92', 'purchase_price': 63.82}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 1, 31), 'product': 'product_55', 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 3, 36), 'product': 'product_81', 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 4, 57), 'product': None, 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 8, 55), 'product': 'product_53', 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 10, 1), 'product': 'product_85', 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 13, 37), 'product': None, 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 15, 45), 'product': 'product_2', 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 20, 44), 'product': None, 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 21, 8), 'product': 'product_76', 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 24, 4), 'product': None, 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 27, 30), 'product': 'product_49', 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 29, 47), 'product': None, 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 32, 42), 'product': 'product_55', 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 36, 10), 'product': 'product_73', 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 37, 22), 'product': None, 'purchase_price': None}
    {'user_id': 4, 'session_id': '4_340855', 'state': 'sale', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 39, 6), 'product': 'product_50', 'purchase_price': 39.62}
    {'user_id': 5, 'session_id': '5_343671', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 1, 8), 'product': 'product_70', 'purchase_price': None}
    {'user_id': 5, 'session_id': '5_343671', 'state': 'sale', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 6, 2), 'product': 'product_1', 'purchase_price': 98.55}
    {'user_id': 6, 'session_id': '6_426799', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 0, 56), 'product': 'product_65', 'purchase_price': None}
    {'user_id': 6, 'session_id': '6_426799', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 2), 'product': 'product_62', 'purchase_price': None}
    {'user_id': 6, 'session_id': '6_426799', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 3, 54), 'product': None, 'purchase_price': None}
    {'user_id': 6, 'session_id': '6_426799', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 7, 4), 'product': 'product_78', 'purchase_price': None}
    {'user_id': 6, 'session_id': '6_426799', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 8, 31), 'product': None, 'purchase_price': None}
    {'user_id': 6, 'session_id': '6_426799', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 9, 38), 'product': 'product_83', 'purchase_price': None}
    {'user_id': 6, 'session_id': '6_426799', 'state': 'entry', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 14, 34), 'product': None, 'purchase_price': None}
    {'user_id': 6, 'session_id': '6_426799', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 15, 42), 'product': 'product_46', 'purchase_price': None}
    {'user_id': 6, 'session_id': '6_426799', 'state': 'entry', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 20, 15), 'product': None, 'purchase_price': None}
    {'user_id': 6, 'session_id': '6_426799', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 20, 32), 'product': 'product_13', 'purchase_price': None}
    {'user_id': 6, 'session_id': '6_426799', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 21, 10), 'product': None, 'purchase_price': None}
    {'user_id': 6, 'session_id': '6_426799', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 25, 37), 'product': 'product_70', 'purchase_price': None}
    {'user_id': 6, 'session_id': '6_426799', 'state': 'sale', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 28, 35), 'product': 'product_2', 'purchase_price': 41.77}
    {'user_id': 7, 'session_id': '7_489664', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 4), 'product': None, 'purchase_price': None}
    {'user_id': 7, 'session_id': '7_489664', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 5, 56), 'product': 'product_84', 'purchase_price': None}
    {'user_id': 7, 'session_id': '7_489664', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 7, 43), 'product': None, 'purchase_price': None}
    {'user_id': 7, 'session_id': '7_489664', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 8, 38), 'product': 'product_19', 'purchase_price': None}
    {'user_id': 7, 'session_id': '7_489664', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 9, 42), 'product': None, 'purchase_price': None}
    {'user_id': 7, 'session_id': '7_489664', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 14, 2), 'product': 'product_1', 'purchase_price': None}
    {'user_id': 7, 'session_id': '7_489664', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 15, 22), 'product': None, 'purchase_price': None}
    {'user_id': 7, 'session_id': '7_489664', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 17, 7), 'product': 'product_41', 'purchase_price': None}
    {'user_id': 7, 'session_id': '7_489664', 'state': 'entry', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 19, 24), 'product': None, 'purchase_price': None}
    {'user_id': 7, 'session_id': '7_489664', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 22, 37), 'product': 'product_83', 'purchase_price': None}
    {'user_id': 7, 'session_id': '7_489664', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 23, 47), 'product': 'product_71', 'purchase_price': None}
    {'user_id': 7, 'session_id': '7_489664', 'state': 'sale', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 28, 34), 'product': 'product_26', 'purchase_price': 73.49}
    {'user_id': 8, 'session_id': '8_995242', 'state': 'product_view', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 0, 56), 'product': 'product_100', 'purchase_price': None}
    {'user_id': 8, 'session_id': '8_995242', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 4, 16), 'product': None, 'purchase_price': None}
    {'user_id': 8, 'session_id': '8_995242', 'state': 'sale', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 4, 47), 'product': 'product_10', 'purchase_price': 32.16}
    {'user_id': 9, 'session_id': '9_598049', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 2, 59), 'product': 'product_36', 'purchase_price': None}
    {'user_id': 9, 'session_id': '9_598049', 'state': 'sale', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 5, 57), 'product': 'product_80', 'purchase_price': 33.1}
    {'user_id': 10, 'session_id': '10_188434', 'state': 'add_to_cart', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 3, 1), 'product': 'product_9', 'purchase_price': None}
    {'user_id': 10, 'session_id': '10_188434', 'state': 'check_in', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 4, 38), 'product': None, 'purchase_price': None}
    {'user_id': 10, 'session_id': '10_188434', 'state': 'sale', 'time_stamp': datetime.datetime(2023, 1, 1, 0, 9, 37), 'product': 'product_48', 'purchase_price': 28.16}

can you use the python faker to create a user genrator and a product generator then incorporate these into the about event_genrator adding support for `units_sold` and the page_address based on the product. The products should have a category and thier price should be drawn from a category level normal distribution

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2023,
  author = {Bochman, Oren},
  title = {Event Generator},
  date = {2023-02-16},
  url = {https://orenbochman.github.io/posts/2024/2023-03-16-events-generator/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2023. “Event Generator.” February 16. <https://orenbochman.github.io/posts/2024/2023-03-16-events-generator/>.
