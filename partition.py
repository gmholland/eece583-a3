import os.path
import random
import logging
import time
import stats
from math import exp
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from colours import colour_pool

class Node:
    """Class representing a circuit element.
    
    Data attributes:
        ID - Node number, determined from benchmark file
        loc - Site of temporary location of cell
        nets - list of nets the node belongs to"""

    def __init__(self, ID):
        self.ID = ID
        self.loc = None
        self.nets = []

    def __str__(self):
        return 'Node_{}({},{})'.format(self.ID, self.loc.row, self.loc.col)

    def get_partial_cost(self):
        """Get HPBB cost for nets that include this node."""
        partial_cost = 0
        for net in self.nets:
            partial_cost += net.get_hpbb()
        return partial_cost


class Net:
    """Class representing a net.
    
    Data attributes:
        nodes - list of nodes the net connects"""

    def __init__(self):
        self.nodes = []
        self.colour = 'black'

    def get_hpbb(self):
        """Get length of half-perimter bounding box of net."""
        if not self.nodes:
            return 0

        # initialize max and mins to first element coords
        x_min = x_max = self.nodes[0].loc.col
        y_min = y_max = self.nodes[0].loc.row

        # find max and min row and col values of remaining nodes
        for node in self.nodes[1:]:
            if node.loc.col < x_min:
                x_min = node.loc.col
            elif node.loc.col > x_max:
                x_max = node.loc.col

            if node.loc.row < y_min:
                y_min = node.loc.row
            elif node.loc.row > y_max:
                y_max = node.loc.row

        # calculate half perimeter bounding box
        # - multiply y coord by 2 to account for empty rows between cells
        hpbb = (x_max - x_min) + 2*(y_max - y_min)
        return hpbb


class Site:
    """Class representing a cell site.
    
    Data attributes:
        row - row number in the cell layout
        col - column number in the cell layout
        content - pointer to Node element if occupied, None otherwise
        text_id - canvas ID of site label, set to the Node number of content
        rect_id - canvas ID of site rectangle
    """

    def __init__(self, row=None, col=None):
        self.row = row
        self.col = col
        self.content = None
        self.text_id = None
        self.rect_id = None

    def __str__(self):
        return '[{}]'.format(self.content)

    def is_empty(self):
        return self.content == None

    def set_text(self, text=''):
        """Set text label of Site."""
        # create canvas text if needed
        if self.text_id == None:
            x, y = self.get_rect_center()
            self.text_id = canvas.create_text(x, y, text=text)
        else:
            canvas.itemconfigure(self.text_id, text=text)

    def update_rect(self):
        """Colour the rectangle according to content, set text to Node ID."""
        if self.is_empty():
            canvas.itemconfigure(self.rect_id, fill='white')
            #self.set_text('') # debugging: put ID label on each node
        else:
            canvas.itemconfigure(self.rect_id, fill='light grey')
            #self.set_text(self.content.ID) # debug: put ID label on each node

    def get_rect_center(self):
        """Returns (x, y) coordinates of center of Site's canvas rectangle."""
        x1, y1, x2, y2 = canvas.coords(self.rect_id) # get rect coords
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return (center_x, center_y)


class Layout:
    """Class representing the chip layout."""

    def __init__(self):
        self.ncells = 0
        self.nconnections = 0
        self.nrows = 0
        self.ncols = 0
        self.nsites = 0
        self.grid = [[]]
        self.netlist = []
        self.nodelist = []

    def init_grid(self, nrows, ncols):
        """Initialize the grid to given size by populating with empty sites."""
        self.grid = [[Site(col=x, row=y) for x in range(ncols)] for y in range(nrows)]
        self.nrows = nrows
        self.ncols = ncols
        self.nsites = nrows * ncols

    def init_nodelist(self, ncells):
        """Initialize the node list by populating with ncells Nodes."""
        self.nodelist = [Node(i) for i in range(self.ncells)]

    def init_netlist(self):
        """Initialize netlist as an empty list."""
        self.netlist = []

    def get_site_by_id(self, site_id):
        """Return site from grid given a site ID."""
        row = site_id // self.ncols
        col = site_id % self.ncols
        return self.grid[row][col]

    def calculate_cost(self):
        """Caclulate cost of current placement."""
        cost = 0
        for net in self.netlist:
            cost = cost + net.get_hpbb()
        return cost

    def parse_netlist(self, filepath):
        """Parse a netlist and populate the grid.
        
        filepath - the full path of the netlist file to parse"""
        with open(filepath, 'r') as f:
            # first line is ncells, nconnections and grid size
            line = f.readline().strip().split()

            # initialize node list
            self.ncells = int(line[0])
            self.init_nodelist(self.ncells)

            self.nconnections = int(line[1])

            # initialize grid of sites
            nrows = int(line[2])
            ncols = int(line[3])
            self.init_grid(nrows, ncols)

            logging.info('ncells={}, nconnections={}, nrows={}, ncols={}, nsites={}'.format(
                self.ncells, self.nconnections, nrows, ncols, self.nsites))

            # next lines describe netlist
            self.init_netlist()

            # randomize net colours
            random.shuffle(colour_pool)

            # parse nets
            for net_id in range(self.nconnections):
                line = f.readline().strip().split()

                # set net colour
                net = Net()
                net.colour = colour_pool[net_id % len(colour_pool)]

                # each line corresponds to a net giving node IDs of the nodes
                #  contained in the net
                ncells_in_net = line[0]
                for i in line[1:]:
                    node_id = int(i)
                    # lookup corresonding node in nodelist
                    node = self.nodelist[node_id]

                    node.nets.append(net)
                    net.nodes.append(node)

                self.netlist.append(net)

        logging.info('parsed {} nets in netlist file'.format(
            len(self.netlist)))

    def print_nodelist(self):
        for node in self.nodelist:
            print(node, end=',')
        print()

    def print_netlist(self):
        for i, net in enumerate(self.netlist):
            print(i, end=':')
            for node in net.nodes:
                print(node, end=',')
            print()

    def print_grid(self):
        for row in self.grid:
            for site in row:
                print(site, end=' ')
            print()


def open_benchmark(*args):
    """Function called when pressing Open button.
    
    Opens a dialog for user to select a netlist file and calls
    Layout.parse_netlist."""

    # open a select file dialog for user to choose a benchmark file
    openfilename = filedialog.askopenfilename()
    # return if user cancels out of dialog
    if not openfilename:
        return

    logging.info("opened benchmark:{}".format(openfilename))
    filename.set(os.path.basename(openfilename))
    layout.parse_netlist(openfilename)

    # enable the Place button
    place_btn.state(['!disabled'])

    # draw layout of empty cells
    init_canvas()


def place(*args):
    """Function called when pressing Place button.

    Performs initial random placement."""

    # first place all nodes randomly
    initialize_placement()

    layout.cost = layout.calculate_cost()

    update_canvas()

    cost = layout.calculate_cost()
    cost_text.set(cost)
    logging.info('initial cost = {}'.format(cost))

    # enable the Anneal button
    anneal_btn.state(['!disabled'])

    # disable the Place button
    place_btn.state(['disabled'])


def initialize_placement():
    """Randomly assign each node in the nodelist to a cell site."""

    # select ncells random site IDs
    rand_site_ids = random.sample(range(layout.nsites), layout.ncells)

    for i, rand_site_id in enumerate(rand_site_ids):
        # assign selected site to next node in nodelist
        site = layout.get_site_by_id(rand_site_id)
        node = layout.nodelist[i]
        site.content = node
        node.loc = site


def anneal(*args):
    """Function called when pressing the Anneal button.
    
    Place the circuit using simulated annealing."""

    # annealing schedule
    temp = get_initial_temp() # starting temperature
    k = 10 # constant for num iterations at each temp
    niterations = int(k * (layout.ncells)**(4/3))

    # print annealing schedule parameters
    print('T0 =', temp)
    print('niterations =', niterations)

    anneal_outer(temp, niterations)


def get_initial_temp(k=20, nmoves=10):
    """Return a good value to use for the initial temperature.

    Based on standard deviation of a number of random moves."""

    costs = stats.Container()
    costs_list = [0 for i in range(nmoves)]
    for i in range(nmoves):
        # randomly select two sites
        site1, site2 = select_sites()

        # calculate cost of move - only need to consider nets that
        # contain the nodes we swapped
        pre_swap_cost = 0
        if not site1.is_empty():
            node1 = site1.content
            pre_swap_cost += node1.get_partial_cost()
        if not site2.is_empty():
            node2 = site2.content
            pre_swap_cost += node2.get_partial_cost()

        swap_sites(site1, site2)

        post_swap_cost = 0
        if not site1.is_empty():
            node1 = site1.content
            post_swap_cost += node1.get_partial_cost()
        if not site2.is_empty():
            node2 = site2.content
            post_swap_cost += node2.get_partial_cost()

        delta_c = post_swap_cost - pre_swap_cost 
        layout.cost += delta_c
        costs.add(layout.cost)
        costs_list[i] = layout.cost
    
    print('costs of {} random moves:'.format(nmoves), costs_list)
    std_dev = costs.get_std_dev()
    print('std_dev of {} random moves is'.format(nmoves), std_dev)
    initial_temp = k * std_dev
    return initial_temp


def anneal_outer(temp, niterations):
    """Outer loop of annealing function.
    
    Run inner loop, reduce temperature, check exit condition.  """

    print("anneal_outer()")
    accepted_costs = stats.Container()
    # run anneal inner loop
    anneal_inner(temp, niterations, accepted_costs)

    # reduce temp
    temp = get_new_temp(temp, accepted_costs)

    # redraw canvas
    update_canvas()

    print("cost =", layout.cost)

    # check exit condition
    if not exit_condition(temp, accepted_costs):
        # exit condition not met, run outer loop again
        root.after(1000, anneal_outer, temp, niterations)
    else:
        # exit condition met, do final steps
        cost = layout.calculate_cost()
        cost_text.set(cost)
        logging.info('final cost = {}'.format(cost))
        print('final cost = {}'.format(cost))


def anneal_inner(temp, niterations, accepted_costs):
    """Inner loop of simulated annealing algorithm."""

    naccepted_moves = 0
    ntotal_moves = 0
    for i in range(niterations):
        # randomly select two sites
        [site1, site2] = select_sites()

        # calculate cost of move - only need to consider nets that
        # contain the nodes we swapped
        pre_swap_cost = 0
        if not site1.is_empty():
            node1 = site1.content
            pre_swap_cost += node1.get_partial_cost()
        if not site2.is_empty():
            node2 = site2.content
            pre_swap_cost += node2.get_partial_cost()

        swap_sites(site1, site2)

        post_swap_cost = 0
        if not site1.is_empty():
            node1 = site1.content
            post_swap_cost += node1.get_partial_cost()
        if not site2.is_empty():
            node2 = site2.content
            post_swap_cost += node2.get_partial_cost()

        delta_c = post_swap_cost - pre_swap_cost 

        # r = random(0, 1)
        r = random.random()

        if r < exp(-delta_c / temp):
            # take move (keep swap)
            layout.cost += delta_c
            accepted_costs.add(layout.cost)
            naccepted_moves += 1
        else:
            # don't take move (undo swap)
            swap_sites(site2, site1)
        ntotal_moves += 1

    accept_rate = 100 * naccepted_moves / ntotal_moves
    print("accept_rate =", accept_rate)


def get_new_temp(temp, accepted_costs):
    """Return the next annealing temperature.
    
    Based on standard deviation of accepted moves at previous temperature."""

    print('get_new_temp()')
    std_dev = accepted_costs.get_std_dev()
    print('std_dev(accepted_costs) =', std_dev)
    new_temp = temp * exp(-0.7 * temp / std_dev)

    # avoid overflow errors due to very low temperatures
    if new_temp < 0.1:
        new_temp = 0.1

    print('new T =', new_temp)
    return new_temp


def exit_condition(temp, accepted_costs):
    """Check annealing exit condition
    
    Based on standard deviation of the costs of accepted moves."""
    
    print("exit_condition()")
    std_dev = accepted_costs.get_std_dev()
    print('std_dev(accepted costs) =', std_dev)
    if std_dev < 2:
        return True
    else:
        return False



def select_sites():
    """Return a list of 2 randomly selected sites, only one may be empty."""
    while True:
        [site_id1, site_id2] = random.sample(range(layout.nsites), 2)
        site1 = layout.get_site_by_id(site_id1)
        site2 = layout.get_site_by_id(site_id2)
    
        # try again if both sites are empty
        if site1.is_empty() and site2.is_empty():
            continue
        else:
            break

    return [site1, site2]


def swap_sites(site1, site2):
    """Swap content of two sites."""
    temp_content = site1.content
    site1.content = site2.content
    site2.content = temp_content

    if site1.content != None:
        site1.content.loc = site1
    if site2.content != None:
        site2.content.loc = site2


# GUI functions
def init_canvas():
    """Initialize the canvas with rectangles for according to layout."""

    # clear canvas
    canvas.delete(ALL)

    # calculate size of each site
    # TODO make rdim scale to size of layout
    if layout.nrows >= 50:
        rdim = 8
    elif layout.nrows >= 30:
        rdim = 10
    elif layout.nrows >= 20:
        rdim = 15
    elif layout.nrows >= 20:
        rdim = 20
    else:
        rdim = 30

    rh = rw = rdim
    xoffset = yoffset = rdim // 5

    # resize the canvas
    cw = layout.ncols * rw + 2 * xoffset
    ch = (2 * layout.nrows - 1) * rh + 2 * yoffset
    canvas.config(width=cw, height=ch)

    # create rectangles for each cell site
    for row in layout.grid:
        for site in row:
            x1 = site.col * rw + xoffset
            x2 = x1 + rw
            y1 = site.row * rh * 2 + yoffset
            y2 = y1 + rh
            site.rect_id = canvas.create_rectangle(x1, y1, x2, y2,
                    fill='white')


def update_canvas():
    """Redraw the canvas and update statistics being displayed."""
    clear_nets()
    update_rects()
    draw_nets()
    cost_text.set(layout.cost)


def clear_nets(*args):
    """Remove nets from canvas."""
    canvas.delete('ratsnest')


def update_rects(*args):
    """Redraw sites in canvas."""
    # Update rectanges of sites
    for row in layout.grid:
        for site in row:
            site.update_rect()


def draw_nets(*args):
    """Draw nets in canvas."""
    # Update rats nest of nets
    for net in layout.netlist:
        source = net.nodes[0]
        x1, y1 = source.loc.get_rect_center()
        for sink in net.nodes[1:]:
            x2, y2 = sink.loc.get_rect_center()
            canvas.create_line(x1, y1, x2, y2, fill=net.colour, tags='ratsnest')


# main function
if __name__ == '__main__':
    # set random number generator seed 
    random.seed(0)

    # setup logfile
    logfilename = 'placer.log'
    logging.basicConfig(filename=logfilename, filemode='w', level=logging.INFO)

    # chip layout
    layout = Layout()

    # setup gui
    root = Tk()
    root.title("Assignment2-Placement")

    # add frames to gui
    top_frame = ttk.Frame(root, padding="3 3 12 12")
    top_frame.grid(column=0, row=0)
    top_frame.columnconfigure(0, weight=1)
    top_frame.rowconfigure(0, weight=1)
    canvas_frame = ttk.Frame(top_frame)
    canvas_frame.grid(column=0, row=0, sticky=(N,E,S,W))
    btn_frame = ttk.Frame(top_frame)
    btn_frame.grid(column=0, row=1)
    stats_frame = ttk.Frame(top_frame)
    stats_frame.grid(column=1, row=0)

    # setup canvas frame (contains benchmark label and canvas)
    filename = StringVar()
    benchmark_lbl = ttk.Label(canvas_frame, textvariable=filename)
    benchmark_lbl.grid(column=0, row=0)
    canvas = Canvas(canvas_frame, width=640, height=480, bg="dark grey")
    canvas.grid(column=0, row=1, padx=5, pady=5)

    # setup button frame (contains buttons)
    open_btn = ttk.Button(btn_frame, text="Open", command=open_benchmark)
    open_btn.grid(column=0, row=0, padx=5, pady=5)
    place_btn = ttk.Button(btn_frame, text="Place", command=place)
    place_btn.grid(column=1, row=0, padx=5, pady=5)
    place_btn.state(['disabled'])
    anneal_btn = ttk.Button(btn_frame, text="Anneal", command=anneal)
    anneal_btn.grid(column=2, row=0, padx=5, pady=5)
    anneal_btn.state(['disabled'])

    # setup stats frame (contains statistics)
    cost_text = StringVar()
    cost_text.set('-')
    ttk.Label(stats_frame, text="cost:").grid(column=1, row=1)
    cost_lbl = ttk.Label(stats_frame, textvariable=cost_text)
    cost_lbl.grid(column=2, row=1)

    # run main event loop for gui
    root.mainloop()
