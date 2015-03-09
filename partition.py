import os.path
import random
import logging
import math
from colours import colour_pool
from bucket_list import BucketList
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

class Node:
    """Class representing a circuit element.
    
    Data attributes:
        ID - Node number, determined from benchmark file
        block_ID - block ID of assigned block in partition
        nets - list of nets the node belongs to
        gain - amount cutsize would decrease if node moved to other block
        text_id - canvas ID of node label, set to the Node ID
        rect_id - canvas ID of node rectangle
    """
    def __init__(self, ID):
        self.ID = ID
        self.block_ID = None
        self.gain = 0
        self._locked = False
        self.nets = []
        self.text_id = None
        self.rect_id = None

    def __str__(self):
        return 'Node_{}(g={}, b={})'.format(self.ID, self.gain, self.block_ID)

    def lock(self):
        self._locked = True

    def unlock(self):
        self._locked = False

    def is_locked(self):
        return self._locked

    def is_unlocked(self):
        return not self._locked

    def adjust_gain(self, adjustment):
        """Modify gain value keeping bucket consistent.

        adjustment - integer amount to add to gain.
        """
        # pull node out of block
        layout.block[self.block_ID].remove_node(self)

        # update gain 
        self.gain += adjustment

        # add node to new position in bucket
        layout.block[self.block_ID].add_unlocked_node(self)

    def set_text(self, text=''):
        """Set text label on node rectangle."""
        # create canvas text if needed
        x, y = self.get_rect_center()
        self.text_id = canvas.create_text(x, y, text=text)

    def get_rect_center(self):
        """Returns (x, y) coordinates of center of Node's canvas rectangle."""
        x1, y1, x2, y2 = canvas.coords(self.rect_id) # get rect coords
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return (center_x, center_y)


class Net:
    """Class representing a net.
    
    Data attributes:
        nodes - list of nodes the net connects
        colour - net colour in GUI
        distribution - list: [# of nodes in block 0, # of nodes in block 1]
    """
    def __init__(self):
        self.nodes = []
        self.colour = 'black'
        self.distribution = [0, 0]

    def init_distribution(self):
        self.distribution = [0, 0]
        for node in self.nodes:
            self.distribution[node.block_ID] += 1

    def is_cut(self):
        """Return True if net is cut, False otherwise."""
        return (self.distribution[0] > 0) and (self.distribution[1] > 0)


class Block:
    """Class represnting a block in the partition."""
    def __init__(self, pmax):
        self._node_ids = set()
        self._bucket = BucketList(pmax)

    def reset(self):
        self._node_ids = set()
        self._bucket.reset()

    def size(self):
        """Return number of nodes in block."""
        return len(self._node_ids)

    def add_unlocked_node(self, node):
        """Add specified unlocked node to the block."""
        if node.is_locked():
            raise ValueError
        self._node_ids.add(node.ID)
        self._bucket.add_node(node)

    def add_locked_node(self, node):
        """Add locked noode to the block."""
        if node.is_unlocked():
            raise ValueError
        self._node_ids.add(node.ID)

    def remove_node(self, node):
        """Remove specified node from the block."""
        self._node_ids.remove(node.ID)
        self._bucket.remove_node(node)

    def has_unlocked_nodes(self):
        """Return True if there are unlocked nodes, False otherwise."""
        return not self._bucket.is_empty()

    def pop_max_gain_node(self):
        """Return node with max gain and lock it."""
        node = self._bucket.pop()
        self._node_ids.remove(node.ID)
        return node

    def get_max_gain(self):
        return self._bucket.get_max_gain()

    def __contains__(self, node):
        """Return True if node is in the block, False otherwise."""
        return node.ID in self._node_ids

    def copy_set(self):
        return self._node_ids.copy()


class Layout:
    """Class representing the chip layout."""

    def __init__(self):
        self.ncells = 0
        self.nconnections = 0
        self.netlist = []
        self.nodelist = []
        self.cutsize = 0

    def init_nodelist(self, ncells):
        """Initialize the node list by populating with ncells Nodes."""
        self.nodelist = [Node(i) for i in range(self.ncells)]

    def init_netlist(self):
        """Initialize netlist as an empty list."""
        self.netlist = []

    def set_net_distribution(self):
        for net in self.netlist:
            net.init_distribution()

    def calculate_cutsize(self):
        """Caclulate cost of current placement."""
        cutsize = 0
        for net in self.netlist:
            if net.is_cut():
                cutsize += 1
        return cutsize

    def parse_netlist(self, filepath):
        """Parse a netlist file to populate the Layout data structures.
        
        filepath - the full path of the netlist file to parse"""
        with open(filepath, 'r') as f:
            # first line is ncells, nconnections and grid size
            line = f.readline().strip().split()

            # initialize node list
            self.ncells = int(line[0])
            self.init_nodelist(self.ncells)

            self.nconnections = int(line[1])

            logging.info('ncells={}, nconnections={}'.format(
                self.ncells, self.nconnections))

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

        # create block data structures and store in list for easy indexing
        pmax = self.get_pmax()
        self.block = [Block(pmax), Block(pmax)]
        logging.info('pmax={}'.format(pmax))

        global iteration
        iteration = 1
        iteration_text.set(iteration)


    def get_pmax(self):
        """Return max(number of pins on node(i) for i in layout.netlist"""
        pmax = 0
        for node in self.nodelist:
            p = len(node.nets)
            if p > pmax:
                pmax = p
        return pmax

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


def open_benchmark(*args):
    """Function called when pressing Open button.
    
    Opens a dialog for user to select a netlist file and calls
    Layout.parse_netlist."""

    # open a select file dialog for user to choose a benchmark file
    openfilename = filedialog.askopenfilename()
    # return if user cancels out of dialog
    if not openfilename:
        return

    # setup logfile
    logfilename = os.path.basename(openfilename) + '.log'
    logging.basicConfig(filename=logfilename, filemode='w', level=logging.INFO)

    logging.info("opened benchmark:{}".format(openfilename))
    filename.set(os.path.basename(openfilename))
    layout.parse_netlist(openfilename)

    # enable the Place button
    partition_btn.state(['!disabled'])

    # initialize canvas
    gui.init_canvas() 


def partition(*args):
    """Function called when pressing Partition button.

    Partitions circuit using Kernighan-Lin."""

    # create random initial partition
    initialize_partition()

    set_initial_gains()

    # get initial cutsize
    layout.cutsize = layout.calculate_cutsize()
    cutsize_text.set(layout.cutsize)
    logging.info('initial cutsize = {}'.format(layout.cutsize))

    # intialize best partition, mincut and prev mincut
    layout.best_partition = save_partition()
    layout.prev_mincut = layout.mincut = layout.cutsize

    partition_btn.state(['disabled'])

    gui.draw_canvas()

    root.after(100, KL_inner)


def initialize_partition():
    """Randomly partition the nodes equally"""

    # Generate a random list of node IDs
    rand_node_IDs = random.sample(range(layout.ncells), layout.ncells)

    for i, node_ID in enumerate(rand_node_IDs):
        node = layout.nodelist[node_ID]
        node.block_ID = i % 2

    # # assign nodes to alternating blocks
    # if layout.ncells % 2 == 0:
    #     midpoint = layout.ncells // 2
    # else:
    #     midpoint = layout.ncells // 2 + 1
    # for node in layout.nodelist:
    #     if node.ID < midpoint:
    #         node.block_ID = 0
    #     else:
    #         node.block_ID = 1


def set_initial_gains():
    """Set initial gain values for each node.
    
    Also does the following:
        - sets net distribution
        - unlocks all nodes
        - adds nodes to blocks based on node.block_ID
    """

    # initialize net distributions
    layout.set_net_distribution()

    # initialize gain on each node
    for node in layout.nodelist:
        node.unlock()
        node.gain = 0
        F = node.block_ID # "from" block ID
        T = (node.block_ID + 1) % 2 # "to" block ID
        for net in node.nets:
            if net.distribution[F] == 1:
                node.gain += 1 # increment gain
            if net.distribution[T] == 0:
                node.gain -= 1 # decrement gain

        # add node to appropriate block
        layout.block[F].add_unlocked_node(node)


def KL_inner():

    # step 2:
    # - select base node (node with max gain)
    base_node = select_base_node()

    # step 3:
    # - move base node to other block, lock it and update gains
    move_node(base_node)

    # update cutsize
    layout.cutsize -= base_node.gain

    # if cutsize is the minimum for this pass, save partition
    if layout.cutsize < layout.mincut:
        layout.mincut = layout.cutsize
        layout.best_partition = save_partition()

    gui.draw_canvas()

    # continue while there are unlocked nodes
    if layout.block[0].has_unlocked_nodes() or layout.block[1].has_unlocked_nodes():
        root.after(1, KL_inner)
    else:
        root.after(1000, KL_reset)


def KL_reset():
    """Reset partition to best seen during pass."""

    global iteration
    reset_saved_partition(layout.best_partition)

    gui.draw_canvas()

    # continue for up to 6 iterations or until mincut stops improving
    if iteration < 6 and layout.mincut != layout.prev_mincut:
        iteration += 1
        iteration_text.set(iteration)
        layout.prev_mincut = layout.mincut
        root.after(1000, KL_inner)
    else:
        # enable buttons
        partition_btn.state(['!disabled'])


def select_base_node():
    """Choose node to move based on gain and balance condition and return it."""
    # if equal number of nodes in each block
    if layout.block[0].size() == layout.block[1].size():
        # choose node with higher gain
        if layout.block[0].get_max_gain() > layout.block[1].get_max_gain():
            b = 0
        elif layout.block[0].get_max_gain() < layout.block[1].get_max_gain():
            b = 1
        else:
            # break tie
            b = random.choice([0, 1])
    # move node from block with more nodes
    elif layout.block[0].size() > layout.block[1].size():
        b = 0
    else:
        b = 1

    # get node to move
    base_node = layout.block[b].pop_max_gain_node()
    return base_node


def move_node(node):
    """Move node to opposite block and update gains."""
    F = node.block_ID # "from" block ID
    T = (node.block_ID + 1) % 2 # "to" block ID
    node.lock()
    node.block_ID = T
    layout.block[T].add_locked_node(node)
    for net in node.nets:
        # check critical nets before the move
        if net.distribution[T] == 0:
            # increment gains of all free nodes on net
            for neighbour in net.nodes:
                if neighbour.is_unlocked():
                    neighbour.adjust_gain(1)

        elif net.distribution[T] == 1:
            # decrement gain of the only T node on net if it is free
            for neighbour in net.nodes:
                if neighbour.is_unlocked() and neighbour.block_ID == T:
                    neighbour.adjust_gain(-1)

        # change net distribution to reflect the move
        net.distribution[F] -= 1
        net.distribution[T] += 1

        # check critical nets after the move
        if net.distribution[F] == 0:
            # decrement gains of all free nodes on net
            for neighbour in net.nodes:
                if neighbour.is_unlocked():
                    neighbour.adjust_gain(-1)
            
        elif net.distribution[F] == 1:
            # increment gain of only F node on net if it is free
            for neighbour in net.nodes:
                if neighbour.is_unlocked() and neighbour.block_ID == F:
                    neighbour.adjust_gain(1)


def save_partition():
    """Return a saved partition.
    
    A saved partition consists of a list of node IDs for each block.
    """
    block0_IDs = list(layout.block[0].copy_set())
    block1_IDs = list(layout.block[1].copy_set())
    return [block0_IDs, block1_IDs]


def reset_saved_partition(partition):
    """Reset the layout data structures to a specific partition.
    
    All nodes are unlocked as a result and block data structures
    are filled appropriately.
    """
    # reset block data structures
    layout.block[0].reset()
    layout.block[1].reset()

    # set block ID for each node
    for block, block_node_IDs in enumerate(partition):
        for node_ID in block_node_IDs:
            node = layout.nodelist[node_ID]
            node.block_ID = block

    # calculate initial gains and in doing so, fill the block data structures
    set_initial_gains()

    # update cutsize
    layout.cutsize = layout.mincut
    cutsize_text.set(layout.cutsize)
    logging.info('iteration {}: best mincut seen = {}'.format(iteration, layout.cutsize))


# GUI functions
class GUI:
    def init_canvas(self):
        """Initialize canvas, set to appropriate size."""
        # clear canvas
        canvas.delete(ALL)

        # clear statistics
        iteration_text.set(iteration)
        cutsize_text.set('-')

        self.rdim = 25 # rectangle dimensions
        self.node_pad = 5 # padding between node rectangles
        self.x_pad = 50 # padding in x coordinate between partitions
        self.y_pad = 10 # padding from top and bottom of canvas

        max_cells_per_block = (layout.ncells // 2) + 2
        if max_cells_per_block > 25:
            self.nrows = math.ceil(math.sqrt(max_cells_per_block))
            self.ncols = math.ceil(max_cells_per_block / self.nrows)
        else:
            self.nrows = max_cells_per_block
            self.ncols = 1
        self.cw = 2 * (2*self.x_pad + self.ncols*self.rdim + (self.ncols - 1)*self.node_pad)
        self.ch = 2*self.y_pad + self.nrows*self.rdim + (self.nrows - 1)*self.node_pad
        canvas.config(width=self.cw, height=self.ch)

    def draw_canvas(self):
        """Draw the canvas and update statistics being displayed."""
        # clear canvas
        canvas.delete(ALL)

        self.draw_nodes()
        self.draw_nets()
        cutsize_text.set(layout.cutsize)

    def draw_nodes(self):
        """Redraw nodes in each block."""
        rh = rw = self.rdim
        x = [self.x_pad, self.cw//2 + self.x_pad]
        y = [self.y_pad, self.y_pad]
        node_count = [0, 0]
        # Draw rectanges for each node
        for node in layout.nodelist:
            # calculate coords of node rectangle
            x1 = x[node.block_ID]
            x2 = x1 + rw
            y1 = y[node.block_ID]
            y2 = y1 + rh

            # create node rectangle
            if node.is_locked():
                fill = 'light grey'
            else:
                fill = 'white'
            node.rect_id = canvas.create_rectangle(x1, y1, x2, y2, fill=fill)
            node_count[node.block_ID] += 1

            # update x, y value for next rectangle
            # - if on last node in row, move to top of next column
            if (node_count[node.block_ID] % self.nrows) == 0:
                x[node.block_ID] = x2 + self.node_pad
                y[node.block_ID] = self.y_pad
            else:
                x[node.block_ID] = x1
                y[node.block_ID] = y2 + self.node_pad


    def draw_nets(self):
        """Draw nets in canvas."""
        # Draw rats nest of nets
        for net in layout.netlist:
            source = net.nodes[0]
            x1, y1 = source.get_rect_center()
            for sink in net.nodes[1:]:
                x2, y2 = sink.get_rect_center()
                canvas.create_line(x1, y1, x2, y2, fill=net.colour)


# main function
if __name__ == '__main__':
    # set random number generator seed 
    random.seed(0)

    # chip layout
    layout = Layout()

    # setup gui
    gui = GUI()
    root = Tk()
    root.title("Assignment3-Partitioning")

    # add frames to gui
    top_frame = ttk.Frame(root, padding="3 3 12 12")
    top_frame.grid(column=0, row=0)
    top_frame.columnconfigure(0, weight=1)
    top_frame.rowconfigure(0, weight=1)
    canvas_frame = ttk.Frame(top_frame)
    canvas_frame.grid(column=0, row=0, sticky=(N,E,S,W))
    stats_frame = ttk.Frame(top_frame)
    stats_frame.grid(column=0, row=1)
    btn_frame = ttk.Frame(top_frame)
    btn_frame.grid(column=0, row=2)

    # setup canvas frame (contains benchmark label and canvas)
    filename = StringVar()
    benchmark_lbl = ttk.Label(canvas_frame, textvariable=filename)
    benchmark_lbl.grid(column=0, row=0)
    canvas = Canvas(canvas_frame, width=320, height=240, bg="dark grey")
    canvas.grid(column=0, row=1, padx=5, pady=5)

    # setup button frame (contains buttons)
    open_btn = ttk.Button(btn_frame, text="Open", command=open_benchmark)
    open_btn.grid(column=0, row=0, padx=5, pady=5)
    partition_btn = ttk.Button(btn_frame, text="Partition", command=partition)
    partition_btn.grid(column=1, row=0, padx=5, pady=5)
    partition_btn.state(['disabled'])

    # setup stats frame (contains statistics)
    cutsize_text = StringVar()
    cutsize_text.set('-')

    iteration = 1
    iteration_text = StringVar()
    iteration_text.set(iteration)

    ttk.Label(stats_frame, text="cutsize:").grid(column=1, row=1, sticky=E)
    cutsize_lbl = ttk.Label(stats_frame, textvariable=cutsize_text)
    cutsize_lbl.grid(column=2, row=1, sticky=W)

    ttk.Label(stats_frame, text="iteration:").grid(column=1, row=2, sticky=E)
    iteration_lbl = ttk.Label(stats_frame, textvariable=iteration_text)
    iteration_lbl.grid(column=2, row=2, sticky=W)

    # run main event loop for gui
    root.mainloop()
