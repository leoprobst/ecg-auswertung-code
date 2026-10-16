"""Generic conduction variants. Named atrial muscle routes are not insulated tracts.

Geometry is an explicit research construction, never patient-specific anatomy.
Sources: PMID 30366158, 15466628, 5070112. Variant frequencies are not inferred.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

VARIANTS = {
    "legacy": "Bisheriges vereinfachtes Netz",
    "septal_main": "Erweitert · Septalfaszikel aus Hauptstamm",
    "septal_anterior": "Erweitert · Septalfaszikel aus anteriorem Faszikel",
    "septal_posterior": "Erweitert · Septalfaszikel aus posteriorem Faszikel",
    "septal_plexus": "Erweitert · septales Verbindungsgeflecht",
    "fossa_connection": "Erweitert · optionale Fossa-Verbindung",
    "dual_av_paths": "Erweitert · zwei generische AV-Zugänge",
    "left_accessory": "Erweitert · akzessorische linksseitige AV-Verbindung",
    "right_accessory": "Erweitert · akzessorische rechtsseitige AV-Verbindung",
}

STRUCTURE_SCOPE = {
    "bachmann": "generic interatrial connection; working atrial myocardium",
    "coronary_sinus_musculature": "optional inferior muscular bridge, not blood in the coronary sinus",
    "posterosuperior_connection": "generic posterior-superior atrial bridge",
    "fossa_ovalis_connection": "optional, not assumed universal",
    "anterior_middle_posterior_atrial_routes": "preferential muscle routes; dedicated internodal cables disputed",
    "crista_terminalis": "generic working-myocardial route",
    "septopulmonary_septoatrial": "generic left-atrial muscle routes",
    "AV_fast_slow": "optional slow-phenotype access paths; no validated human dual-node kinetics",
    "His_RBB_LBB_anterior_posterior_septal": "explicit generic paths; septal origin configurable",
    "Purkinje_arborization": "sparse terminal connections, not complete patient arborization",
    "Kent_like_AV": "optional generic bypass; no validated accessory pathway localization",
    "Mahaim_nodofascicular_fasciculoventricular": "catalogued only; dedicated validated pathways not implemented",
    "SA_exit_paths": "single generic exit; detailed sinus node exit anatomy not implemented",
}


def extend_network(tissue, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown conduction architecture")
    if variant == "legacy":
        return tissue
    from .model import Tissue
    xyz, cells, regions, fibers = (list(a) for a in (tissue.xyz, tissue.cells, tissue.regions, tissue.fibers))
    edges = list(map(tuple, tissue.edges))
    branches = {k: v.copy() for k, v in tissue.branches.items()}
    sites = {k: v.copy() for k, v in tissue.sites.items()}

    def nearest(nodes, point):
        return int(nodes[np.argmin(np.linalg.norm(np.asarray(xyz)[nodes] - point, axis=1))])

    def cable(name, a, b, cell):
        start, end = np.asarray(xyz[a]), np.asarray(xyz[b])
        distance = np.linalg.norm(end-start)
        if distance < 1e-6:
            raise ValueError("Coincident pathway endpoints")
        points = np.linspace(start, end, max(2, int(np.ceil(distance/2))+1))[1:-1]
        ids = [a] + list(range(len(xyz), len(xyz)+len(points))) + [b]
        xyz.extend(points)
        cells.extend([cell]*len(points))
        regions.extend([0]*len(points))
        fibers.extend(np.tile((end-start)/distance, (len(points), 1)))
        edges.extend(zip(ids[:-1], ids[1:]))
        branches[name] = np.array(ids)

    def route(name, node_set, a, b):
        node_set = np.asarray(node_set)
        allowed = np.zeros(len(tissue.xyz), bool)
        allowed[node_set] = True
        e = tissue.edges[allowed[tissue.edges].all(axis=1)]
        distance = np.linalg.norm(tissue.xyz[e[:, 0]]-tissue.xyz[e[:, 1]], axis=1)
        graph = coo_matrix((np.r_[distance, distance], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                           shape=(len(tissue.xyz), len(tissue.xyz))).tocsr()
        _, parent = dijkstra(graph, directed=False, indices=a, return_predecessors=True)
        chain, current = [b], b
        for _ in range(len(node_set)):
            if current == a:
                branches[name] = np.array(chain[::-1])
                return
            current = int(parent[current])
            if current < 0:
                break
            chain.append(current)
        raise ValueError("Requested atrial muscle route disconnected")

    ra, la = sites["ra_wall_all"], sites["la_wall_all"]
    ra_xyz, la_xyz = tissue.xyz[ra], tissue.xyz[la]
    ra_low = int(ra[np.argmin(ra_xyz[:, 2])])
    la_low = int(la[np.argmin(la_xyz[:, 2])])
    ra_back = int(ra[np.argmax(ra_xyz[:, 1])])
    la_back = int(la[np.argmax(la_xyz[:, 1])])
    ra_high = int(sites["sa_node"][0])
    la_front = int(sites["la_bachmann"][0])
    ra_av = int(sites["ra_av_access"][0])
    old_bridge = tuple(sorted((ra_high, la_front)))
    edges = [e for e in edges if tuple(sorted(e)) != old_bridge]
    cable("bachmann", ra_high, la_front, 3)
    cable("coronary_sinus_musculature", ra_low, la_low, 3)
    cable("posterosuperior_connection", ra_back, la_back, 3)
    route("atrial_anterior_route", ra, ra_high, ra_av)
    route("middle_proximal", ra, ra_high, ra_back)
    route("middle_distal", ra, ra_back, ra_av)
    branches["atrial_middle_route"] = np.unique(np.r_[branches.pop("middle_proximal"), branches.pop("middle_distal")])
    route("crista_terminalis_route", ra, ra_high, ra_low)
    route("atrial_posterior_route", ra, ra_low, ra_av)
    route("septopulmonary_route", la, la_front, la_back)
    route("septoatrial_route", la, la_back, la_low)
    if variant == "fossa_connection":
        a = nearest(ra, (ra_xyz.mean(0)+la_xyz.mean(0))/2)
        b = nearest(la, (ra_xyz.mean(0)+la_xyz.mean(0))/2)
        cable("fossa_ovalis_connection", a, b, 3)
    parents = ["lbb_anterior", "lbb_posterior"] if variant == "septal_plexus" else [
        {"septal_anterior": "lbb_anterior", "septal_posterior": "lbb_posterior"}.get(variant, "lbb")]
    target = int(sites["lv_mid_septal"][0])
    for parent in parents:
        mid = (tissue.xyz[target]+tissue.xyz[sites["his"][0]])/2
        cable("lbb_septal_from_"+parent, nearest(branches[parent], mid), target, 5)
    if variant == "dual_av_paths":
        cable("av_slow_access", ra_low, int(sites["his"][0]), 4)
        branches["av_fast_access"] = np.r_[ra_av, branches["his"]]
    if variant in {"left_accessory", "right_accessory"}:
        left = variant == "left_accessory"
        target = int(sites["lv_lateral_base" if left else "rv_free_wall_base"][0])
        cable("accessory_av_connection", nearest(la if left else ra, tissue.xyz[target]), target, 3)
    return Tissue(np.asarray(xyz), np.unique(np.sort(np.asarray(edges), axis=1), axis=0), np.asarray(cells),
                  np.asarray(regions), np.asarray(fibers), sites, branches, "generic_research_network_"+variant)
