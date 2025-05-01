float deltaR(float eta1, float phi1, float eta2, float phi2) {
    float deta = eta1 - eta2;
    float dphi = phi1 - phi2;
    
    while (dphi > M_PI) dphi -= 2*M_PI;
    while (dphi < -M_PI) dphi += 2*M_PI;
 
    return sqrt(deta*deta + dphi*dphi);
}


void printProgressBar(int done, int total, int barWidth = 80) {
    float progress = (float)done / total;
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "#";
        else
            std::cout << " ";
    }
    std::cout << "] (" << done << "/" << total << ") " << int(progress * 100.0) << "%\r";
    std::cout.flush();

    // Add a newline when progress reaches 100%
    if (done == total) {
        std::cout << std::endl;
    }
}


void do_stuff_raw(string filepath, string histpath, unsigned int N_ENTRIES_MAX) {
    std::cout << "Computing dRs in " << filepath << std::endl;

    bool do_parts = true;
    bool do_topos = true;
    bool do_cells = false;
    
    // read file
    TFile *file = TFile::Open(filepath.c_str());

    TTree *tree = (TTree*)file->Get("Out_Tree");
    
    unsigned int nentries = tree->GetEntries();
    if (nentries > N_ENTRIES_MAX)
        nentries = N_ENTRIES_MAX;

    // read the branches (cells)
    std::vector<float> *cell_eta = nullptr;
    std::vector<float> *cell_phi = nullptr;
    std::vector<float> *cell_e = nullptr;
    std::vector<int> *cell_topo_idx = nullptr;
    std::vector<int> *cell_parent_idx = nullptr;
    if (do_cells) {
        tree->SetBranchAddress("cell_eta", &cell_eta);
        tree->SetBranchAddress("cell_phi", &cell_phi);
        tree->SetBranchAddress("cell_e", &cell_e);
        tree->SetBranchAddress("cell_topo_idx", &cell_topo_idx);
        tree->SetBranchAddress("cell_parent_idx", &cell_parent_idx);
    }

    // read the branches (topos)
    std::vector<float> *topo_eta = nullptr;
    std::vector<float> *topo_phi = nullptr;
    std::vector<float> *topo_e = nullptr;
    tree->SetBranchAddress("topo_bary_eta", &topo_eta);
    tree->SetBranchAddress("topo_bary_phi", &topo_phi);
    tree->SetBranchAddress("topo_e", &topo_e);

    // read the branches (particles)
    std::vector<float> *particle_eta = nullptr;
    std::vector<float> *particle_phi = nullptr;
    std::vector<float> *particle_e = nullptr;
    tree->SetBranchAddress("particle_eta", &particle_eta);
    tree->SetBranchAddress("particle_phi", &particle_phi);
    tree->SetBranchAddress("particle_e", &particle_e);


    // write to file
    TFile *out_file = new TFile(histpath.c_str(), "RECREATE");

    int n_bins = 10000;

    TH1F *h_topo_to_topo_dR = new TH1F("h_topo_to_topo_dR", "h_topo_to_topo_dR", n_bins, 0, 8);
    TH1F *h_topo_to_topo_dR_weighted = new TH1F("h_topo_to_topo_dR_weighted", "h_topo_to_topo_dR_weighted", n_bins, 0, 8);

    TH1F *h_part_to_part_dR = new TH1F("h_part_to_part_dR", "h_part_to_part_dR", n_bins, 0, 8);
    TH1F *h_part_to_part_dR_weighted = new TH1F("h_part_to_part_dR_weighted", "h_part_to_part_dR_weighted", n_bins, 0, 8);

    TH1F *h_cell_to_cell_dR = new TH1F("h_cell_to_cell_dR", "h_cell_to_cell_dR", n_bins, 0, 8);
    TH1F *h_cell_to_cell_dR_weighted = new TH1F("h_cell_to_cell_dR_weighted", "h_cell_to_cell_dR_weighted", n_bins, 0, 8);

    TH1F *h_cell_to_cell_dR_within_topo = new TH1F("h_cell_to_cell_dR_within_topo", "h_cell_to_cell_dR_within_topo", n_bins, 0, 8);
    TH1F *h_cell_to_cell_dR_weighted_within_topo = new TH1F("h_cell_to_cell_dR_weighted_within_topo", "h_cell_to_cell_dR_weighted_within_topo", n_bins, 0, 8);

    TH1F *h_cell_to_cell_dR_within_topo_no_noisy_cell = new TH1F("h_cell_to_cell_dR_within_topo_no_noisy_cell", "h_cell_to_cell_dR_within_topo_no_noisy_cell", n_bins, 0, 8);
    TH1F *h_cell_to_cell_dR_weighted_within_topo_no_noisy_cell = new TH1F("h_cell_to_cell_dR_weighted_within_topo_no_noisy_cell", "h_cell_to_cell_dR_weighted_within_topo_no_noisy_cell", n_bins, 0, 8);

    // loop over events
    for (unsigned int ev_i=0; ev_i<nentries; ev_i++) {
        tree->GetEntry(ev_i);
        printProgressBar(ev_i + 1, nentries);

        if(do_parts) {
            // part to part dR
            for (unsigned int i=0; i<particle_eta->size(); i++) {
                float eta1 = particle_eta->at(i);
                float phi1 = particle_phi->at(i);
                float e1 = particle_e->at(i);

                for (unsigned int j=0; j<particle_eta->size(); j++) {
                    if (i == j) continue;

                    float eta2 = particle_eta->at(j);
                    float phi2 = particle_phi->at(j);
                    float e2 = particle_e->at(j);

                    float dR = deltaR(eta1, phi1, eta2, phi2);

                    h_part_to_part_dR->Fill(dR);
                    h_part_to_part_dR_weighted->Fill(dR, e1 * e2 * 1e-6);
                }
            }
        }

        if(do_topos) {
            // topo to topo dR
            for (unsigned int i=0; i<topo_eta->size(); i++) {
                float eta1 = topo_eta->at(i);
                float phi1 = topo_phi->at(i);
                float e1 = topo_e->at(i);

                for (unsigned int j=0; j<topo_eta->size(); j++) {
                    if (i == j) continue;

                    float eta2 = topo_eta->at(j);
                    float phi2 = topo_phi->at(j);
                    float e2 = topo_e->at(j);

                    float dR = deltaR(eta1, phi1, eta2, phi2);

                    h_topo_to_topo_dR->Fill(dR);
                    h_topo_to_topo_dR_weighted->Fill(dR, e1 * e2 * 1e-6);
                }
            }
        }

        if(do_cells) {

            // cell to cell dR
            for (unsigned int i=0; i<cell_eta->size(); i++) {
                float eta1 = cell_eta->at(i);
                float phi1 = cell_phi->at(i);
                float e1 = cell_e->at(i);

                for (unsigned int j=0; j<cell_eta->size(); j++) {
                    if (i == j) continue;

                    float eta2 = cell_eta->at(j);
                    float phi2 = cell_phi->at(j);
                    float e2 = cell_e->at(j);

                    float dR = deltaR(eta1, phi1, eta2, phi2);

                    h_cell_to_cell_dR->Fill(dR);
                    h_cell_to_cell_dR_weighted->Fill(dR, e1 * e2 * 1e-6);
                }
            }

            // cell to cell dR within topo
            for (unsigned int i=0; i<cell_eta->size(); i++) {
                float eta1 = cell_eta->at(i);
                float phi1 = cell_phi->at(i);
                float e1 = cell_e->at(i);

                for (unsigned int j=0; j<cell_eta->size(); j++) {
                    if (i == j) continue;

                    if (cell_topo_idx->at(i) != cell_topo_idx->at(j)) continue;

                    float eta2 = cell_eta->at(j);
                    float phi2 = cell_phi->at(j);
                    float e2 = cell_e->at(j);

                    float dR = deltaR(eta1, phi1, eta2, phi2);

                    if (dR < 0.4) {
                        h_cell_to_cell_dR_within_topo->Fill(dR);
                        h_cell_to_cell_dR_weighted_within_topo->Fill(dR, e1 * e2 * 1e-6);
                    }
                }
            }

            // cell to cell dR within topo (no noisy cell)
            for (unsigned int i=0; i<cell_eta->size(); i++) {
                float eta1 = cell_eta->at(i);
                float phi1 = cell_phi->at(i);
                float e1 = cell_e->at(i);

                for (unsigned int j=0; j<cell_eta->size(); j++) {
                    if (i == j) continue;

                    if (cell_topo_idx->at(i) != cell_topo_idx->at(j)) continue;

                    if ((cell_parent_idx->at(i) == -1) || (cell_parent_idx->at(j) == -1)) continue;

                    float eta2 = cell_eta->at(j);
                    float phi2 = cell_phi->at(j);
                    float e2 = cell_e->at(j);

                    float dR = deltaR(eta1, phi1, eta2, phi2);

                    if (dR < 0.4) {
                        h_cell_to_cell_dR_within_topo_no_noisy_cell->Fill(dR);
                        h_cell_to_cell_dR_weighted_within_topo_no_noisy_cell->Fill(dR, e1 * e2 * 1e-6);
                    }
                }
            }
        }
    }

    // cd
    out_file->cd();

    // write the histograms to file
    h_topo_to_topo_dR->Write();
    h_topo_to_topo_dR_weighted->Write();

    h_part_to_part_dR->Write();
    h_part_to_part_dR_weighted->Write();

    if (do_cells){
        h_cell_to_cell_dR->Write();
        h_cell_to_cell_dR_weighted->Write();

        h_cell_to_cell_dR_within_topo->Write();
        h_cell_to_cell_dR_weighted_within_topo->Write();

        h_cell_to_cell_dR_within_topo_no_noisy_cell->Write();
        h_cell_to_cell_dR_weighted_within_topo_no_noisy_cell->Write();
    }

    // close the files
    out_file->Close();
}



void do_stuff_seg(string filepath, string histpath, unsigned int N_ENTRIES_MAX) {
    std::cout << "Computing dRs in " << filepath << std::endl;

    bool do_parts = true;
    bool do_topos = true;
    
    // read file
    TFile *file = TFile::Open(filepath.c_str());
    TTree *tree = (TTree*)file->Get("EventTree");
    
    int32_t eventNumber = 0;
    tree->SetBranchAddress("eventNumber", &eventNumber);

    unsigned int n_segenergtries = 0;
    std::set<int32_t> uniqueEvents;
    for (unsigned int i = 0; i < tree->GetEntries(); ++i) {
        tree->GetEntry(i);
        uniqueEvents.insert(eventNumber);
        if (uniqueEvents.size() > N_ENTRIES_MAX) {
            break;
        }
        n_segenergtries++;
    }

    // read the branches (topos)
    int ntopo = 0;
    float topo_eta[1000] = {0};
    float topo_phi[1000] = {0};
    float topo_e[1000] = {0};
    tree->SetBranchAddress("ntopo", &ntopo);
    tree->SetBranchAddress("topo_eta", &topo_eta);
    tree->SetBranchAddress("topo_phi", &topo_phi);
    tree->SetBranchAddress("topo_e", &topo_e);

    // read the branches (particles)
    int nparticle = 0;
    float particle_eta[1000] = {0};
    float particle_phi[1000] = {0};
    float particle_e[1000] = {0};
    tree->SetBranchAddress("nparticle", &nparticle);
    tree->SetBranchAddress("particle_eta", &particle_eta);
    tree->SetBranchAddress("particle_phi", &particle_phi);
    tree->SetBranchAddress("particle_e", &particle_e);


    // write to file
    TFile *out_file = new TFile(histpath.c_str(), "RECREATE");

    int n_bins = 10000;

    TH1F *h_topo_to_topo_dR = new TH1F("h_topo_to_topo_dR", "h_topo_to_topo_dR", n_bins, 0, 8);
    TH1F *h_topo_to_topo_dR_weighted = new TH1F("h_topo_to_topo_dR_weighted", "h_topo_to_topo_dR_weighted", n_bins, 0, 8);

    TH1F *h_part_to_part_dR = new TH1F("h_part_to_part_dR", "h_part_to_part_dR", n_bins, 0, 8);
    TH1F *h_part_to_part_dR_weighted = new TH1F("h_part_to_part_dR_weighted", "h_part_to_part_dR_weighted", n_bins, 0, 8);

    // loop over events
    for (unsigned int ev_i=0; ev_i<n_segenergtries; ev_i++) {
        tree->GetEntry(ev_i);
        printProgressBar(ev_i + 1, n_segenergtries);

        if(do_parts) {
            // part to part dR
            for (unsigned int i=0; i<nparticle; i++) {
                float eta1 = particle_eta[i];
                float phi1 = particle_phi[i];
                float e1 = particle_e[i];

                for (unsigned int j=0; j<nparticle; j++) {
                    if (i == j) continue;

                    float eta2 = particle_eta[j];
                    float phi2 = particle_phi[j];
                    float e2 = particle_e[j];

                    float dR = deltaR(eta1, phi1, eta2, phi2);

                    h_part_to_part_dR->Fill(dR);
                    h_part_to_part_dR_weighted->Fill(dR, e1 * e2 * 1e-6);
                }
            }
        }

        if(do_topos) {
            // topo to topo dR
            for (unsigned int i=0; i<ntopo; i++) {
                float eta1 = topo_eta[i];
                float phi1 = topo_phi[i];
                float e1 = topo_e[i];

                for (unsigned int j=0; j<ntopo; j++) {
                    if (i == j) continue;

                    float eta2 = topo_eta[j];
                    float phi2 = topo_phi[j];
                    float e2 = topo_e[j];

                    float dR = deltaR(eta1, phi1, eta2, phi2);

                    h_topo_to_topo_dR->Fill(dR);
                    h_topo_to_topo_dR_weighted->Fill(dR, e1 * e2 * 1e-6);
                }
            }
        }
    }

    // cd
    out_file->cd();

    // write the histograms to file
    h_topo_to_topo_dR->Write();
    h_topo_to_topo_dR_weighted->Write();

    h_part_to_part_dR->Write();
    h_part_to_part_dR_weighted->Write();

    // close the files
    out_file->Close();
}






int correlation() {
    string filepath, histpath;

    // dijet
    filepath = "path to dijet.root";
    histpath = "...";
    do_stuff_raw(filepath, histpath, 10000);

    filepath = "path to dijet_seg_bw0.4.root";
    histpath = "...";
    do_stuff_seg(filepath, histpath, 10000);


    // ZHbb
    filepath = "path to ZHbb_boosted.root";
    histpath = "...";
    do_stuff_raw(filepath, histpath, 10000);

    filepath = "path to ZHbb_boosted_seg_bw0.4.root";
    histpath = "...";
    do_stuff_seg(filepath, histpath, 10000);


    // ttbar 
    filepath = "path to ttbar.root";
    histpath = "...";
    do_stuff_raw(filepath, histpath, 10000);

    filepath = "path to ttbar_seg_bw0.4.root";
    histpath = "...";
    do_stuff_seg(filepath, histpath, 10000);

    return 0;
}