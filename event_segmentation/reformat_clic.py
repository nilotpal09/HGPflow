import numpy as np
import awkward
import uproot
import vector
import glob
import os
import sys
import multiprocessing
from scipy.sparse import coo_matrix
import argparse
from tqdm import tqdm

# include hgpflow in the PYTHONPATH, since we don't have it as a package yet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility.tree_writer import TreeWriter

track_coll = "SiTracks_Refitted"
mc_coll = "MCParticles"

# the feature matrices will be saved in this order
particle_feature_order = ["PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy"]

# arrange track and cluster features such that pt (et), eta, phi, p (energy) are in the same spot
# so we can easily use them in skip connections
track_feature_order = [
    "elemtype",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "p",
    "chi2",
    "ndf",
    "dEdx",
    "dEdxError",
    "radiusOfInnermostHit",
    "tanLambda",
    "D0",
    "omega",
    "Z0",
    "time",
]
cluster_feature_order = [
    "elemtype",
    "et",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "position.x",
    "position.y",
    "position.z",
    "iTheta",
    "energy_ecal",
    "energy_hcal",
    "energy_other",
    "num_hits",
    "sigma_x",
    "sigma_y",
    "sigma_z",
]
hit_feature_order = [
    "elemtype",
    "et",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "position.x",
    "position.y",
    "position.z",
    "time",
    "subdetector",
    "type",
]


def track_pt(omega):
    a = 3 * 10**-4
    b = 4  # B-field in tesla, from clicRec_e4h_input

    return a * np.abs(b / omega)


def map_pdgid_to_candid(pdgid, charge):
    if pdgid == 0:
        return 0

    # photon, electron, muon
    if pdgid in [22, 11, 13]:
        return pdgid

    # charged hadron
    if abs(charge) > 0:
        return 211

    # neutral hadron
    return 130


def map_charged_to_neutral(pdg):
    if pdg == 0:
        return 0
    if pdg == 11 or pdg == 22:
        return 22
    return 130


def map_neutral_to_charged(pdg):
    if pdg == 130 or pdg == 22:
        return 211
    return pdg


def sanitize(arr):
    arr[np.isnan(arr)] = 0.0
    arr[np.isinf(arr)] = 0.0


class EventData:
    def __init__(
        self,
        gen_features,
        hit_features,
        cluster_features,
        track_features,
        genparticle_to_hit,
        genparticle_to_track,
        hit_to_cluster,
        gp_merges,
    ):
        self.gen_features = gen_features  # feature matrix of the genparticles
        self.hit_features = hit_features  # feature matrix of the calo hits
        self.cluster_features = cluster_features  # feature matrix of the calo clusters
        self.track_features = track_features  # feature matrix of the tracks
        self.genparticle_to_hit = genparticle_to_hit  # sparse COO matrix of genparticles to hits (idx_gp, idx_hit, weight)
        self.genparticle_to_track = (
            genparticle_to_track  # sparse COO matrix of genparticles to tracks (idx_gp, idx_track, weight)
        )
        self.hit_to_cluster = hit_to_cluster  # sparse COO matrix of hits to clusters (idx_hit, idx_cluster, weight)
        self.gp_merges = gp_merges  # sparse COO matrix of any merged genparticles

        self.genparticle_to_hit = (
            np.array(self.genparticle_to_hit[0]),
            np.array(self.genparticle_to_hit[1]),
            np.array(self.genparticle_to_hit[2]),
        )
        self.genparticle_to_track = (
            np.array(self.genparticle_to_track[0]),
            np.array(self.genparticle_to_track[1]),
            np.array(self.genparticle_to_track[2]),
        )
        self.hit_to_cluster = (
            np.array(self.hit_to_cluster[0]),
            np.array(self.hit_to_cluster[1]),
            np.array(self.hit_to_cluster[2]),
        )
        self.gp_merges = np.array(self.gp_merges[0]), np.array(self.gp_merges[1])


def hits_to_features(hit_data, iev, coll, feats):
    feat_arr = {f: hit_data[coll + "." + f][iev] for f in feats}

    # set the subdetector type
    sdcoll = "subdetector"
    feat_arr[sdcoll] = np.zeros(len(feat_arr["type"]), dtype=np.int32)
    if coll.startswith("ECALBarrel"):
        feat_arr[sdcoll][:] = 0
    elif coll.startswith("ECALEndcap"):
        feat_arr[sdcoll][:] = 1
    elif coll.startswith("ECALOther"):
        feat_arr[sdcoll][:] = 2
    elif coll.startswith("HCALBarrel"):
        feat_arr[sdcoll][:] = 3
    elif coll.startswith("HCALEndcap"):
        feat_arr[sdcoll][:] = 4
    elif coll.startswith("HCALOther"):
        feat_arr[sdcoll][:] = 5
    elif coll.startswith("MUON"):
        feat_arr[sdcoll][:] = 6
    else:
        print(coll)
        raise Exception("Unknown subdetector")

    # hit elemtype is always 2
    feat_arr["elemtype"] = 2 * np.ones(len(feat_arr["type"]), dtype=np.int32)

    # precompute some approximate et, eta, phi
    # pos_mag = np.sqrt(feat_arr["position.x"] ** 2 + feat_arr["position.y"] ** 2 + feat_arr["position.z"] ** 2)
    # px = (feat_arr["position.x"] / pos_mag) * feat_arr["energy"]
    # py = (feat_arr["position.y"] / pos_mag) * feat_arr["energy"]
    # pz = (feat_arr["position.z"] / pos_mag) * feat_arr["energy"]
    # feat_arr["et"] = np.sqrt(px**2 + py**2)

    # feat_arr["eta"] = 0.5 * np.log((feat_arr["energy"] + pz) / (feat_arr["energy"] - pz))
    # feat_arr["sin_phi"] = py / feat_arr["energy"]
    # feat_arr["cos_phi"] = px / feat_arr["energy"]
    # feat_arr["phi"] = np.arctan2(feat_arr["sin_phi"], feat_arr["cos_phi"])

    # NILOTPAL: let's compute eta-phi with only the position (no energy)
    r = np.sqrt(feat_arr["position.x"]**2 + feat_arr["position.y"]**2)
    feat_arr["eta"] = -np.log(np.tan(np.arctan2(r, feat_arr["position.z"]) / 2.0))
    feat_arr["phi"] = np.arctan2(feat_arr["position.y"], feat_arr["position.x"])
    feat_arr["sin_phi"] = np.sin(feat_arr["phi"])
    feat_arr["cos_phi"] = np.cos(feat_arr["phi"])
    feat_arr["rho"] = r

    return awkward.Record(feat_arr)


def get_calohit_matrix_and_genadj(hit_data, calohit_links, iev, collectionIDs):
    feats = ["type", "cellID", "energy", "energyError", "time", "position.x", "position.y", "position.z"]

    hit_idx_global = 0
    hit_idx_global_to_local = {}
    hit_feature_matrix = []
    for col in sorted(hit_data.keys()):
        icol = collectionIDs[col]
        hit_features = hits_to_features(hit_data[col], iev, col, feats)
        hit_feature_matrix.append(hit_features)
        for ihit in range(len(hit_data[col][col + ".energy"][iev])):
            hit_idx_global_to_local[hit_idx_global] = (icol, ihit)
            hit_idx_global += 1
    hit_idx_local_to_global = {v: k for k, v in hit_idx_global_to_local.items()}
    hit_feature_matrix = awkward.Record(
        {
            k: awkward.concatenate([hit_feature_matrix[i][k] for i in range(len(hit_feature_matrix))])
            for k in hit_feature_matrix[0].fields
        }
    )

    # add all edges from genparticle to calohit
    calohit_to_gen_weight = calohit_links["CalohitMCTruthLink"]["CalohitMCTruthLink.weight"][iev]
    calohit_to_gen_calo_colid = calohit_links["CalohitMCTruthLink#0"]["CalohitMCTruthLink#0.collectionID"][iev]
    calohit_to_gen_gen_colid = calohit_links["CalohitMCTruthLink#1"]["CalohitMCTruthLink#1.collectionID"][iev]
    calohit_to_gen_calo_idx = calohit_links["CalohitMCTruthLink#0"]["CalohitMCTruthLink#0.index"][iev]
    calohit_to_gen_gen_idx = calohit_links["CalohitMCTruthLink#1"]["CalohitMCTruthLink#1.index"][iev]
    genparticle_to_hit_matrix_coo0 = []
    genparticle_to_hit_matrix_coo1 = []
    genparticle_to_hit_matrix_w = []
    for calo_colid, calo_idx, gen_colid, gen_idx, w in zip(
        calohit_to_gen_calo_colid,
        calohit_to_gen_calo_idx,
        calohit_to_gen_gen_colid,
        calohit_to_gen_gen_idx,
        calohit_to_gen_weight,
    ):
        genparticle_to_hit_matrix_coo0.append(gen_idx)
        genparticle_to_hit_matrix_coo1.append(hit_idx_local_to_global[(calo_colid, calo_idx)])
        genparticle_to_hit_matrix_w.append(w)

    return (
        hit_feature_matrix,
        (genparticle_to_hit_matrix_coo0, genparticle_to_hit_matrix_coo1, genparticle_to_hit_matrix_w),
        hit_idx_local_to_global,
    )


def hit_cluster_adj(prop_data, hit_idx_local_to_global, iev):
    coll_arr = prop_data["PandoraClusters#1"]["PandoraClusters#1.collectionID"][iev]
    idx_arr = prop_data["PandoraClusters#1"]["PandoraClusters#1.index"][iev]
    hits_begin = prop_data["PandoraClusters"]["PandoraClusters.hits_begin"][iev]
    hits_end = prop_data["PandoraClusters"]["PandoraClusters.hits_end"][iev]

    # index in the array of all hits
    hit_to_cluster_matrix_coo0 = []
    # index in the cluster array
    hit_to_cluster_matrix_coo1 = []

    # weight
    hit_to_cluster_matrix_w = []

    # loop over all clusters
    for icluster in range(len(hits_begin)):

        # get the slice in the hit array corresponding to this cluster
        hbeg = hits_begin[icluster]
        hend = hits_end[icluster]
        idx_range = idx_arr[hbeg:hend]
        coll_range = coll_arr[hbeg:hend]

        # add edges from hit to cluster
        for icol, idx in zip(coll_range, idx_range):
            hit_to_cluster_matrix_coo0.append(hit_idx_local_to_global[(icol, idx)])
            hit_to_cluster_matrix_coo1.append(icluster)
            hit_to_cluster_matrix_w.append(1.0)
    return hit_to_cluster_matrix_coo0, hit_to_cluster_matrix_coo1, hit_to_cluster_matrix_w

def get_particle_parent_idx(particle_dict):

    thresh = 1e-8

    path_length = np.sqrt(
        (particle_dict["endpoint_x"] - particle_dict["vertex_x"])**2 +
        (particle_dict["endpoint_y"] - particle_dict["vertex_y"])**2 +
        (particle_dict["endpoint_z"] - particle_dict["vertex_z"])**2
    )

    r_e = np.sqrt(particle_dict["endpoint_x"]**2 + particle_dict["endpoint_y"]**2 + particle_dict["endpoint_z"]**2)
    r_v = np.sqrt(particle_dict["vertex_x"]**2 + particle_dict["vertex_y"]**2 + particle_dict["vertex_z"]**2)

    # require: 
    #    - particles with generatorStatus in [0,1,2] and 
    #    - particles that have an endpoint farther away than its vertex (some photons had endpoint at origin) and
    #    - particles which travel farther than 1mm (not optimized) or are neutral pions
    mask = (particle_dict["generatorStatus"] < 3) & (r_v < r_e) & ( (path_length > 1) | (particle_dict["PDG"] == 111))

    x_v = particle_dict["vertex_x"].to_numpy()[mask].reshape(-1, 1)
    y_v = particle_dict["vertex_y"].to_numpy()[mask].reshape(-1, 1)
    z_v = particle_dict["vertex_z"].to_numpy()[mask].reshape(-1, 1)
    i_v = particle_dict["idx"][mask]

    x_e = particle_dict["endpoint_x"].to_numpy()[mask].reshape(1, -1)
    y_e = particle_dict["endpoint_y"].to_numpy()[mask].reshape(1, -1)
    z_e = particle_dict["endpoint_z"].to_numpy()[mask].reshape(1, -1)
    j_e = particle_dict["idx"][mask]

    dx =  x_v - x_e
    dy =  y_v - y_e
    dz =  z_v - z_e

    d = np.sqrt(dx**2 + dy**2 + dz**2)

    # make the diagonal elements large, in case the particle is its own parent
    d += np.eye(d.shape[0]) * 9999

    matches = (d < thresh)

    # check that each particle has 0 or 1 parent
    # assert np.sum(matches, axis=1).max() < 2, f"Some particles have more than one parent: {np.sum(matches, axis=1)}"
    if len(np.sum(matches, axis=1)) > 0:
        if np.sum(matches, axis=1).max() > 1:
            N = (np.sum(matches, axis=1) > 1).sum()
            print(f"Warning: {N} particles have more than one parent! Can lead to unexpected behavior...")

    child_idx_masked, parent_idx_masked = np.where(matches)

    # map the masked indices back to the original indices
    parent_idx = np.ones_like(particle_dict["idx"]) * -9999
    parent_idx[i_v[child_idx_masked]] = j_e[parent_idx_masked]

    return parent_idx


def gen_to_features(prop_data, iev):
    gen_arr = prop_data[mc_coll][iev]
    gen_arr = {k.replace(mc_coll + ".", ""): gen_arr[k] for k in gen_arr.fields}

    MCParticles_p4 = vector.awk(
        awkward.zip(
            {"mass": gen_arr["mass"], "x": gen_arr["momentum.x"], "y": gen_arr["momentum.y"], "z": gen_arr["momentum.z"]}
        )
    )
    gen_arr["pt"] = MCParticles_p4.pt
    gen_arr["eta"] = MCParticles_p4.eta
    gen_arr["phi"] = MCParticles_p4.phi
    gen_arr["energy"] = MCParticles_p4.energy
    gen_arr["mass"] = MCParticles_p4.mass
    gen_arr["sin_phi"] = np.sin(gen_arr["phi"])
    gen_arr["cos_phi"] = np.cos(gen_arr["phi"])
    gen_arr["idx"] = np.arange(len(gen_arr["PDG"]))

    return awkward.Record(
        {
            "PDG": gen_arr["PDG"],
            "generatorStatus": gen_arr["generatorStatus"],
            "simulatorStatus": gen_arr["simulatorStatus"],
            "charge": gen_arr["charge"],
            "pt": gen_arr["pt"],
            "eta": gen_arr["eta"],
            "phi": gen_arr["phi"],
            "sin_phi": gen_arr["sin_phi"],
            "cos_phi": gen_arr["cos_phi"],
            "energy": gen_arr["energy"],
            "mass": gen_arr["mass"],
            "vertex_x": gen_arr["vertex.x"],
            "vertex_y": gen_arr["vertex.y"],
            "vertex_z": gen_arr["vertex.z"],
            "endpoint_x": gen_arr["endpoint.x"],
            "endpoint_y": gen_arr["endpoint.y"],
            "endpoint_z": gen_arr["endpoint.z"],
            "parents_begin": gen_arr["parents_begin"],
            "parents_end": gen_arr["parents_end"],
            "daughters_begin": gen_arr["daughters_begin"],
            "daughters_end": gen_arr["daughters_end"]
        }
    )


def genparticle_track_adj(sitrack_links, iev):
    trk_to_gen_trkidx = sitrack_links["SiTracksMCTruthLink#0"]["SiTracksMCTruthLink#0.index"][iev]
    trk_to_gen_genidx = sitrack_links["SiTracksMCTruthLink#1"]["SiTracksMCTruthLink#1.index"][iev]
    trk_to_gen_w = sitrack_links["SiTracksMCTruthLink"]["SiTracksMCTruthLink.weight"][iev]

    genparticle_to_track_matrix_coo0 = awkward.to_numpy(trk_to_gen_genidx)
    genparticle_to_track_matrix_coo1 = awkward.to_numpy(trk_to_gen_trkidx)
    genparticle_to_track_matrix_w = awkward.to_numpy(trk_to_gen_w)

    return genparticle_to_track_matrix_coo0, genparticle_to_track_matrix_coo1, genparticle_to_track_matrix_w


def cluster_to_features(prop_data, hit_features, hit_to_cluster, iev):
    cluster_arr = prop_data["PandoraClusters"][iev]
    feats = ["type", "position.x", "position.y", "position.z", "iTheta", "phi", "energy"]
    ret = {feat: cluster_arr["PandoraClusters." + feat] for feat in feats}

    # NILOTPAL: eta and phi from Pandora seems strange, so we recompute it
    r = np.sqrt(ret["position.x"]**2 + ret["position.y"]**2)
    ret["eta"] = -np.log(np.tan(np.arctan2(r, ret["position.z"]) / 2.0))
    ret["phi"] = np.arctan2(ret["position.y"], ret["position.x"])
    ret["rho"] = r
    ret["sin_phi"] = np.sin(ret["phi"])
    ret["cos_phi"] = np.cos(ret["phi"])


    hit_idx = np.array(hit_to_cluster[0])
    cluster_idx = np.array(hit_to_cluster[1])
    cl_energy_ecal = []
    cl_energy_hcal = []
    cl_energy_other = []
    num_hits = []

    cl_sigma_x = []
    cl_sigma_y = []
    cl_sigma_z = []

    cl_sigma_eta = []
    cl_sigma_phi = []
    cl_sigma_rho = []

    n_cl = len(ret["energy"])
    for cl in range(n_cl):
        msk_cl = cluster_idx == cl
        hits = hit_idx[msk_cl]

        num_hits.append(len(hits))

        subdets = hit_features["subdetector"][hits]

        hits_energy = hit_features["energy"][hits]

        hits_posx = hit_features["position.x"][hits]
        hits_posy = hit_features["position.y"][hits]
        hits_posz = hit_features["position.z"][hits]

        energy_ecal = np.sum(hits_energy[subdets <= 2]) # 0, 1, 2 are ecal
        energy_hcal = np.sum(hits_energy[(subdets >= 3) & (subdets <= 5)]) # 3, 4, 5 are hcal
        energy_other = np.sum(hits_energy[subdets > 5]) # 6 is muon

        cl_energy_ecal.append(energy_ecal)
        cl_energy_hcal.append(energy_hcal)
        cl_energy_other.append(energy_other)

        cl_sigma_x.append(np.std(hits_posx))
        cl_sigma_y.append(np.std(hits_posy))
        cl_sigma_z.append(np.std(hits_posz))

        # new stuff
        hits_eta = hit_features["eta"][hits]
        hits_phi = hit_features["phi"][hits]
        hits_sin_phi = hit_features["sin_phi"][hits]
        hits_cos_phi = hit_features["cos_phi"][hits]
        hits_rho = hit_features["rho"][hits]

        cl_sigma_eta.append(np.std(hits_eta))
        mean_cos_phi = np.mean(hits_cos_phi); mean_sin_phi = np.mean(hits_sin_phi)
        mean_phi = np.arctan2(mean_sin_phi, mean_cos_phi)
        diff_phi = (hits_phi - mean_phi + np.pi) % (2 * np.pi) - np.pi # shift to [-pi, pi]
        cl_sigma_phi.append(np.sqrt(np.mean(diff_phi**2)))

        cl_sigma_rho.append(np.std(hits_rho))


    ret["energy_ecal"] = np.array(cl_energy_ecal)
    ret["energy_hcal"] = np.array(cl_energy_hcal)
    ret["energy_other"] = np.array(cl_energy_other)
    ret["num_hits"] = np.array(num_hits)
    ret["sigma_x"] = np.array(cl_sigma_x)
    ret["sigma_y"] = np.array(cl_sigma_y)
    ret["sigma_z"] = np.array(cl_sigma_z)

    ret["sigma_eta"] = np.array(cl_sigma_eta)
    ret["sigma_phi"] = np.array(cl_sigma_phi)
    ret["sigma_rho"] = np.array(cl_sigma_rho)


    # NILOTPAL: eta and phi computed from iTheta and phi seems strange
    # tt = awkward.to_numpy(np.tan(ret["iTheta"] / 2.0))
    # eta = awkward.to_numpy(-np.log(tt, where=tt > 0))
    # eta[tt <= 0] = 0.0
    # ret["eta"] = eta


    # costheta = np.cos(ret["iTheta"])
    # ez = ret["energy"] * costheta
    # ret["et"] = np.sqrt(ret["energy"] ** 2 - ez**2)

    # # cluster is always type 2
    # ret["elemtype"] = 2 * np.ones(n_cl, dtype=np.float32)

    # ret["sin_phi"] = np.sin(ret["phi"])
    # ret["cos_phi"] = np.cos(ret["phi"])

    return awkward.Record(ret)


def get_radius(pt_gev, omega, B=4.0):
    q = np.sign(omega)
    R = pt_gev / (0.299792 * q * B)
    return R


def track_helix(pt_GeV, phi_0, q, tan_lambda, phi_max):
    R = get_radius(pt_GeV, q)

    phi_max = - q * phi_max

    x =  R * (np.cos(phi_0 + phi_max + np.pi/2) - np.cos(phi_0 + np.pi/2))
    y =  R * (np.sin(phi_0 + phi_max + np.pi/2) - np.sin(phi_0 + np.pi/2))
    z = -R * tan_lambda * phi_max

    return x, y, z


def get_track_intersection_vect(pt_GeV, phi_0, omega, tan_lambda, max_radius_m=1.502, max_z_m=2.309):
    R  = get_radius(pt_GeV, omega) # meters

    cos_phi_T = 1 - (max_radius_m**2 / (2 * R**2))
    phi_T = np.zeros_like(cos_phi_T)
    phi_T[np.abs(cos_phi_T) <= 1] = np.arccos(cos_phi_T[np.abs(cos_phi_T) <= 1])
    phi_T[np.abs(cos_phi_T) > 1] = np.nan

    phi_z = max_z_m / (R * tan_lambda)

    # R is signed radius
    # case 1: 2*|R| < max_radius_m => goes through endcap (looper)
    # case 2: 2*|R| > max_radius_m (will touch the barrel at least by phi=pi)
    #   |phi_z| > pi => goes through barrel
    #   |phi_z| < pi
    #       |phi_T| < |phi_z| => goes through barrel
    #       |phi_T| > |phi_z| => goes through endcap # below mask is on this

    is_barrel_intersect = np.ones_like(phi_z, dtype=bool) # everything goes through barrel
    is_barrel_intersect[2*np.abs(R) < max_radius_m] = False # if 2*|R| < max_radius_m, it goes through endcap
    mask = (np.abs(phi_z) < np.pi) * (2*np.abs(R) > max_radius_m) * (np.abs(phi_T) > np.abs(phi_z))
    is_barrel_intersect[mask] = False

    phi_max = phi_z
    phi_max[is_barrel_intersect] = phi_T[is_barrel_intersect]
    
    phi_max = np.abs(phi_max)

    x_int, y_int, z_int = track_helix(pt_GeV, phi_0, np.sign(omega), tan_lambda, phi_max)

    theta_int = np.arctan2(np.sqrt(x_int**2 + y_int**2), z_int)
    eta_int   = -np.log(np.clip(np.tan(theta_int/2.0), 1e-8, None))
    phi_int   = np.arctan2(y_int, x_int)

    # extra_info = {
    #     'R': R,
    #     'phi_T': phi_T,
    #     'phi_z': phi_z,
    #     'phi_max': phi_max,
    #     'is_barrel_intersect': is_barrel_intersect,
    # }

    return x_int, y_int, z_int, eta_int, phi_int #, extra_info


def track_to_features(prop_data, iev):
    track_arr = prop_data[track_coll][iev]
    feats_from_track = ["type", "chi2", "ndf", "dEdx", "dEdxError", "radiusOfInnermostHit"]
    ret = {feat: track_arr[track_coll + "." + feat] for feat in feats_from_track}
    n_tr = len(ret["type"])

    # get the index of the first track state
    trackstate_idx = prop_data[track_coll][track_coll + ".trackStates_begin"][iev]
    # get the properties of the track at the first track state (at the origin)
    for k in ["tanLambda", "D0", "phi", "omega", "Z0", "time"]:
        ret[k] = awkward.to_numpy(prop_data["SiTracks_1"]["SiTracks_1." + k][iev][trackstate_idx])

    ret["pt"] = awkward.to_numpy(track_pt(ret["omega"]))
    ret["px"] = awkward.to_numpy(np.cos(ret["phi"])) * ret["pt"]
    ret["py"] = awkward.to_numpy(np.sin(ret["phi"])) * ret["pt"]
    ret["pz"] = awkward.to_numpy(ret["tanLambda"]) * ret["pt"]
    ret["p"] = np.sqrt(ret["px"] ** 2 + ret["py"] ** 2 + ret["pz"] ** 2)
    cos_theta = np.divide(ret["pz"], ret["p"], where=ret["p"] > 0)
    theta = np.arccos(cos_theta)
    tt = np.tan(theta / 2.0)
    eta = awkward.to_numpy(-np.log(tt, where=tt > 0))
    eta[tt <= 0] = 0.0
    ret["eta"] = eta

    ret["sin_phi"] = np.sin(ret["phi"])
    ret["cos_phi"] = np.cos(ret["phi"])

    # # track is always type 1
    # ret["elemtype"] = 1 * np.ones(n_tr, dtype=np.float32)

    ### track intersection with calorimeter 
    x_int, y_int, z_int, eta_int, phi_int = get_track_intersection_vect(
        ret["pt"], ret["phi"], ret["omega"], ret["tanLambda"]
    )

    ret["x_int"] = x_int
    ret["y_int"] = y_int
    ret["z_int"] = z_int
    ret["eta_int"] = eta_int
    ret["phi_int"] = phi_int

    return awkward.Record(ret)


def filter_adj(adj, all_to_filtered):
    i0s_new = []
    i1s_new = []
    ws_new = []
    for i0, i1, w in zip(*adj):
        if i0 in all_to_filtered:
            i0_new = all_to_filtered[i0]
            i0s_new.append(i0_new)
            i1s_new.append(i1)
            ws_new.append(w)
    return np.array(i0s_new), np.array(i1s_new), np.array(ws_new)


def get_genparticles_and_adjacencies(prop_data, hit_data, calohit_links, sitrack_links, iev, collectionIDs):
    gen_features = gen_to_features(prop_data, iev)
    hit_features, genparticle_to_hit, hit_idx_local_to_global = get_calohit_matrix_and_genadj(
        hit_data, calohit_links, iev, collectionIDs
    )
    hit_to_cluster = hit_cluster_adj(prop_data, hit_idx_local_to_global, iev)
    cluster_features = cluster_to_features(prop_data, hit_features, hit_to_cluster, iev)
    track_features = track_to_features(prop_data, iev)
    genparticle_to_track = genparticle_track_adj(sitrack_links, iev)

    n_gp = awkward.count(gen_features["PDG"])
    n_track = awkward.count(track_features["type"])
    n_hit = awkward.count(hit_features["type"])
    n_cluster = awkward.count(cluster_features["type"])

    if len(genparticle_to_track[0]) > 0:
        gp_to_track = (
            coo_matrix((genparticle_to_track[2], (genparticle_to_track[0], genparticle_to_track[1])), shape=(n_gp, n_track))
            .max(axis=1)
            .todense()
        )
    else:
        gp_to_track = np.zeros((n_gp, 1))

    gp_to_calohit = coo_matrix((genparticle_to_hit[2], (genparticle_to_hit[0], genparticle_to_hit[1])), shape=(n_gp, n_hit))
    calohit_to_cluster = coo_matrix((hit_to_cluster[2], (hit_to_cluster[0], hit_to_cluster[1])), shape=(n_hit, n_cluster))
    gp_to_cluster = (gp_to_calohit * calohit_to_cluster).sum(axis=1)

    # 60% of the hits of a track must come from the genparticle
    gp_in_tracker = np.array(gp_to_track >= 0.6)[:, 0]

    # at least 10% of the energy of the genparticle should be matched to a calorimeter cluster
    gp_in_calo = (np.array(gp_to_cluster)[:, 0] / gen_features["energy"]) > 0.01 # EXPERIMENTAL: moved from 10% to 1%

    gp_interacted_with_detector = gp_in_tracker | gp_in_calo

    definition = 'truth'

    if definition == 'truth':
        mask_visible = (
            (gen_features["generatorStatus"]<3) & 
            (np.abs(gen_features["PDG"])!=12) & 
            (np.abs(gen_features["PDG"])!=14) & 
            (np.abs(gen_features["PDG"])!=16) & 
            (gen_features["energy"]>0.01) &
            (np.abs(gen_features["eta"]) < 4)
        )
    elif definition == 'target':
        mask_visible = (gen_features["energy"] > 0.01) & gp_interacted_with_detector
    else:
        raise ValueError('Unknown definition')
    # print("gps total={} visible={}".format(n_gp, np.sum(mask_visible)))
    idx_all_masked = np.where(mask_visible)[0]
    genpart_idx_all_to_filtered = {idx_all: idx_filtered for idx_filtered, idx_all in enumerate(idx_all_masked)}

    dict_visible = {feat: gen_features[feat][mask_visible] for feat in gen_features.fields}
    dict_visible["interacted"] = gp_interacted_with_detector[mask_visible]
    dict_visible["idx"]        = np.arange(len(dict_visible["PDG"]))
    dict_visible["parent_idx"] = get_particle_parent_idx(dict_visible)

    gen_features = awkward.Record(dict_visible)

    genparticle_to_hit = filter_adj(genparticle_to_hit, genpart_idx_all_to_filtered)
    genparticle_to_track = filter_adj(genparticle_to_track, genpart_idx_all_to_filtered)

    return EventData(
        gen_features,
        hit_features,
        cluster_features,
        track_features,
        genparticle_to_hit,
        genparticle_to_track,
        hit_to_cluster,
        ([], []),
    )




def get_reco_properties(prop_data, iev):
    reco_arr = prop_data["MergedRecoParticles"][iev]
    reco_arr = {k.replace("MergedRecoParticles.", ""): reco_arr[k] for k in reco_arr.fields}

    reco_p4 = vector.awk(
        awkward.zip(
            {"mass": reco_arr["mass"], "x": reco_arr["momentum.x"], "y": reco_arr["momentum.y"], "z": reco_arr["momentum.z"]}
        )
    )
    reco_arr["pt"] = reco_p4.pt
    reco_arr["eta"] = reco_p4.eta
    reco_arr["phi"] = reco_p4.phi
    reco_arr["energy"] = reco_p4.energy

    msk = reco_arr["type"] != 0
    reco_arr = awkward.Record({k: reco_arr[k][msk] for k in reco_arr.keys()})
    return reco_arr


def process_one_file(fn, ofn, chunk_size, estart=0, estop=None):

    # # output exists, do not recreate
    # if os.path.isfile(ofn):
    #     print("{} exists".format(ofn))
    #     return

    fi = uproot.open(fn)

    arrs = fi["events"]

    if (estop is None) or (estop >  arrs.num_entries):
        estop = arrs.num_entries

    collectionIDs = {
        k: v
        for k, v in zip(
            fi.get("metadata").arrays("CollectionIDs")["CollectionIDs"]["m_names"][0],
            fi.get("metadata").arrays("CollectionIDs")["CollectionIDs"]["m_collectionIDs"][0],
        )
    }

    prop_data = arrs.arrays(
        [
            mc_coll,
            track_coll,
            "SiTracks_1",
            "PandoraClusters",
            "PandoraClusters#1",
            # "PandoraClusters#0",
            "MergedRecoParticles",
        ], entry_start=estart, entry_stop=estop
    )
    calohit_links = arrs.arrays(["CalohitMCTruthLink", "CalohitMCTruthLink#0", "CalohitMCTruthLink#1"], entry_start=estart, entry_stop=estop)
    sitrack_links = arrs.arrays(["SiTracksMCTruthLink", "SiTracksMCTruthLink#0", "SiTracksMCTruthLink#1"], entry_start=estart, entry_stop=estop)

    hit_data = {
        "ECALBarrel": arrs["ECALBarrel"].array(entry_start=estart, entry_stop=estop),
        "ECALEndcap": arrs["ECALEndcap"].array(entry_start=estart, entry_stop=estop),
        "ECALOther": arrs["ECALOther"].array(entry_start=estart, entry_stop=estop),
        "HCALBarrel": arrs["HCALBarrel"].array(entry_start=estart, entry_stop=estop),
        "HCALEndcap": arrs["HCALEndcap"].array(entry_start=estart, entry_stop=estop),
        "HCALOther": arrs["HCALOther"].array(entry_start=estart, entry_stop=estop),
        "MUON": arrs["MUON"].array(entry_start=estart, entry_stop=estop),
    }

    ret = []


    out_obj = TreeWriter(ofn, "events", chunk_size=chunk_size)

    for iev in tqdm(range(estop - estart)):

        # get the reco particles
        reco_arr = get_reco_properties(prop_data, iev)
        reco_type = np.abs(reco_arr["type"])
        n_rps = len(reco_type)
        reco_features = awkward.Record(
            {
                "PDG": np.abs(reco_type),
                "charge": reco_arr["charge"],
                "pt": reco_arr["pt"],
                "eta": reco_arr["eta"],
                "sin_phi": np.sin(reco_arr["phi"]),
                "cos_phi": np.cos(reco_arr["phi"]),
                "energy": reco_arr["energy"],
            }
        )

        # get the genparticles and the links between genparticles and tracks/clusters
        gpdata = get_genparticles_and_adjacencies(prop_data, hit_data, calohit_links, sitrack_links, iev, collectionIDs)
        dict_ = vars(gpdata)
        
        n_particles = len(dict_['gen_features'].PDG)
        n_clusters  = len(dict_['cluster_features'].energy)


        # track to particle match
        gp_to_track_start = np.array(dict_['genparticle_to_track'][0])
        gp_to_track_end   = np.array(dict_['genparticle_to_track'][1])
        gp_to_track_wt    = np.array(dict_['genparticle_to_track'][2])

        particle_track_idx = np.zeros(n_particles) - 9999
        track_particle_idx = []
        for tr_i in range(len(dict_['track_features'].pt)):

            # find the matched particle idx
            mask = gp_to_track_end == tr_i
            if mask.sum() == 1:
                p_idx = gp_to_track_start[mask][0]
            elif mask.sum() > 1:
                p_idx = gp_to_track_start[mask][np.argmax(gp_to_track_wt[mask])]
            else:
                p_idx = -9999

            if p_idx != -9999: # if particle found
                particle_track_idx[p_idx] = tr_i
                track_particle_idx.append(p_idx)
            else:
                track_particle_idx.append(-9999)

        track_particle_idx = np.array(track_particle_idx)

        # track_particle_idx can have duplicates
        dup_tpi, dup_count = np.unique(track_particle_idx, return_counts=True)
        non_minus_one_mask = dup_tpi != -9999
        dup_tpi   = dup_tpi[non_minus_one_mask]
        dup_count = dup_count[non_minus_one_mask]

        if len(dup_tpi) > 0:
            if max(dup_count) > 1:
                dup_p_idx = dup_tpi[np.where(dup_count > 1)[0]]

                # find the best track for each duplicate
                for p_idx in dup_p_idx:
                    matched_tr_idxs = np.where(track_particle_idx == p_idx)[0]
                    deta = dict_['track_features'].eta[matched_tr_idxs] - dict_['gen_features'].eta[p_idx]
                    dphi = dict_['track_features'].phi[matched_tr_idxs] - dict_['gen_features'].phi[p_idx]
                    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi # shift to [-pi, pi]
                    dr_sq = deta**2 + dphi**2
                    dpt  = dict_['track_features'].pt[matched_tr_idxs] - dict_['gen_features'].pt[p_idx]

                    metric = np.sqrt(dr_sq + (dpt/dict_['gen_features'].pt[p_idx])**2)
                    best_tr_idx = matched_tr_idxs[np.argmin(metric)]

                    # remove all other tracks
                    track_particle_idx[matched_tr_idxs] = -9999
                    track_particle_idx[best_tr_idx] = p_idx

                    particle_track_idx[p_idx] = best_tr_idx


        # cell to topocluster matching
        hit_to_cluster_start = np.array(dict_['hit_to_cluster'][0])
        hit_to_cluster_end   = np.array(dict_['hit_to_cluster'][1])
        hit_to_cluster_wt    = np.array(dict_['hit_to_cluster'][2]) 

        # cell_idx      = hit_to_cluster_start[np.argsort(hit_to_cluster_start)]
        cell_topo_idx = hit_to_cluster_end[np.argsort(hit_to_cluster_start)]
        # cell_topo_wt  = hit_to_cluster_wt[np.argsort(hit_to_cluster_start)]


        # topocluster to particle matching
        gp_to_hit_start = np.array(dict_['genparticle_to_hit'][0])
        gp_to_hit_end   = np.array(dict_['genparticle_to_hit'][1])
        gp_to_hit_wt    = np.array(dict_['genparticle_to_hit'][2])

        # replace the cell idxs with cluster idxs
        gp_to_cluster_start = gp_to_hit_start
        gp_to_cluster_end   = np.zeros(len(gp_to_hit_end)) - 9999
        for i, idx in enumerate(gp_to_hit_end):
            if np.isin(idx, hit_to_cluster_start):
                pos = np.where(idx == hit_to_cluster_start)[0]
                gp_to_cluster_end[i] = hit_to_cluster_end[pos]

        # shrink to shape (n_particles, n_clusters) by binning
        bins = (np.arange(-0.5, n_particles), np.arange(-0.5, n_clusters))
        hs, xs, ys = np.histogram2d(gp_to_cluster_start, gp_to_cluster_end, bins=bins, weights=gp_to_hit_wt)

        # we store only the nonzero elements
        particle_topo_start, particle_topo_end = np.where(hs != 0)
        particle_topo_wt = hs[np.where(hs != 0)]

        # wrong one
        # modify weights so we disregard hits that aren't associated with any cluster (i.e. those with cluster index = -9999)
        # gp_to_hit_wt_mod = np.zeros_like(gp_to_hit_wt)
        # cluster_exists_mask = gp_to_cluster_end != -9999
        # gp_to_hit_wt_mod[cluster_exists_mask] = gp_to_hit_wt[cluster_exists_mask]
        # particle_dep_e = np.histogram(gp_to_cluster_start, bins=np.arange(-0.5, n_particles), weights=gp_to_hit_wt)[0] # gp_to_hit_wt_mod)[0]


        # correct one
        particle_dep_e = np.histogram(particle_topo_start, bins=np.arange(-0.5, n_particles), weights=particle_topo_wt)[0]

        # get phi from sin and cos
        cell_in_topo_mask = np.isin(np.arange(len(dict_['hit_features'].energy)), hit_to_cluster_start)
        pandora_phi = np.arctan2(reco_features.sin_phi, reco_features.cos_phi)

        event_dict = {
            'particle_pt' : dict_['gen_features'].pt,
            'particle_eta': dict_['gen_features'].eta,
            'particle_phi': dict_['gen_features'].phi,
            'particle_e'  : dict_['gen_features'].energy,
            'particle_m'  : dict_['gen_features'].mass,
            'particle_pdg': dict_['gen_features'].PDG,
            'particle_gen_status': dict_['gen_features'].generatorStatus,
            'particle_sim_status': dict_['gen_features'].simulatorStatus,
            'particle_dep_e': particle_dep_e,
            'particle_track_idx': particle_track_idx,

            'particle_vertex_x': dict_['gen_features'].vertex_x,
            'particle_vertex_y': dict_['gen_features'].vertex_y,
            'particle_vertex_z': dict_['gen_features'].vertex_z,
            'particle_endpoint_x': dict_['gen_features'].endpoint_x,
            'particle_endpoint_y': dict_['gen_features'].endpoint_y,
            'particle_endpoint_z': dict_['gen_features'].endpoint_z,
            # 'particle_parents_begin': dict_['gen_features'].parents_begin, # buggy
            # 'particle_parents_end': dict_['gen_features'].parents_end, # buggy
            # 'particle_daughters_begin': dict_['gen_features'].daughters_begin, # buggy
            # 'particle_daughters_end': dict_['gen_features'].daughters_end, # buggy
            'particle_interacted': dict_['gen_features'].interacted,
            'particle_idx': dict_['gen_features'].idx,
            'particle_parent_idx': dict_['gen_features'].parent_idx,

            'track_pt'    : dict_['track_features'].pt,
            'track_eta'   : dict_['track_features'].eta,
            'track_phi'   : dict_['track_features'].phi,
            'track_p'     : dict_['track_features'].p,
            'track_d0'    : dict_['track_features'].D0,
            'track_z0'    : dict_['track_features'].Z0,
            'track_chi2'  : dict_['track_features'].chi2,
            'track_ndf'   : dict_['track_features'].ndf,
            'track_dedx'  : dict_['track_features'].dEdx,
            'track_dedx_error': dict_['track_features'].dEdxError,
            'track_radiusofinnermosthit': dict_['track_features'].radiusOfInnermostHit,
            'track_tanlambda': dict_['track_features'].tanLambda,
            'track_omega'  : dict_['track_features'].omega,
            'track_time'   : dict_['track_features'].time,

            'track_x_int'  : dict_['track_features'].x_int,
            'track_y_int'  : dict_['track_features'].y_int,
            'track_z_int'  : dict_['track_features'].z_int,
            'track_eta_int': dict_['track_features'].eta_int,
            'track_phi_int': dict_['track_features'].phi_int,

            'track_particle_idx': track_particle_idx,
            # 'track_particle_wt' : track_particle_wt,

            'cell_eta'    : dict_['hit_features'].eta[cell_in_topo_mask],
            'cell_phi'    : dict_['hit_features'].phi[cell_in_topo_mask],
            'cell_rho'    : dict_['hit_features'].rho[cell_in_topo_mask],
            'cell_e'      : dict_['hit_features'].energy[cell_in_topo_mask],
            'cell_x'      : dict_['hit_features']['position.x'][cell_in_topo_mask],
            'cell_y'      : dict_['hit_features']['position.y'][cell_in_topo_mask],
            'cell_z'      : dict_['hit_features']['position.z'][cell_in_topo_mask],
            'cell_subdet' : dict_['hit_features'].subdetector[cell_in_topo_mask],

            'cell_to_topo_idx': cell_topo_idx,

            'topo_x'      : dict_['cluster_features']['position.x'],
            'topo_y'      : dict_['cluster_features']['position.y'],
            'topo_z'      : dict_['cluster_features']['position.z'],

            'topo_eta'    : dict_['cluster_features'].eta,
            'topo_phi'    : dict_['cluster_features'].phi,
            'topo_rho'    : dict_['cluster_features'].rho,
            'topo_e'      : dict_['cluster_features'].energy,
            # 'topo_et'     : dict_['cluster_features'].et,
            'topo_num_cells': dict_['cluster_features'].num_hits,
            'topo_sigma_eta': dict_['cluster_features'].sigma_eta,
            'topo_sigma_phi': dict_['cluster_features'].sigma_phi,
            'topo_sigma_rho': dict_['cluster_features'].sigma_rho,
            'topo_energy_ecal': dict_['cluster_features'].energy_ecal,
            'topo_energy_hcal': dict_['cluster_features'].energy_hcal,
            'topo_energy_other': dict_['cluster_features'].energy_other,

            'particle_topo_start': particle_topo_start,
            'particle_topo_end'  : particle_topo_end,
            'particle_topo_wt'   : particle_topo_wt,

            'pandora_pt'  : reco_features.pt,
            'pandora_eta' : reco_features.eta,
            'pandora_phi' : pandora_phi,
            'pandora_e'   : reco_features.energy,
            'pandora_pdg' : reco_features.PDG,
            'pandora_charge': reco_features.charge,
        }
        out_obj.fill_one_ev(event_dict)
        
    out_obj.write()
    out_obj.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='input file', required=True)
    parser.add_argument('--output', '-o', type=str, help='output file', required=True)
    parser.add_argument('--entry_start', '-estart', type=int, required=False, help='uproot entry_start', default=0)
    parser.add_argument('--entry_stop', '-estop', type=int, required=False, help='uproot entry_stopt', default=None)
    parser.add_argument('--chunk_size', '-c', type=int, help='chunk size', default=1000)
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    chunk_size = args.chunk_size

    process_one_file(input_file, output_file, chunk_size, estart=args.entry_start, estop=args.entry_stop)

