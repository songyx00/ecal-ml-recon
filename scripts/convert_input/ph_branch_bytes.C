// scripts/convert_input/ph_branch_bytes_overlay_multi.C
//
// Merge results from MULTIPLE input ROOT files:
// - For each file's TTree, loop all mod*_ph branches.
// - Read branch-level byte counters (TotBytes/ZipBytes) WITHOUT decompressing.
// - Classify EMPTY vs NONEMPTY using Length$(branch) at entry 0 (decompresses only that branch).
// - Accumulate across files, then draw density-normalized overlays in one figure.
// - Also save per-category 1D histos and a combined Zip vs Tot 2D.
//
// Usage examples (ROOT CLI):
//   // Single file (same as previous behavior):
//   root -l -q 'scripts/convert_input/ph_branch_bytes_overlay_multi.C("my.root","tree","merged_bytes.root",true)'
//
//   // Multiple files (newline-separated list text file):
//   root -l -q 'scripts/convert_input/ph_branch_bytes_overlay_multi.C("filelist.txt","tree","merged_bytes.root",true)'
//
//   // Or pass a comma-separated list in one string:
//   root -l -q 'scripts/convert_input/ph_branch_bytes_overlay_multi.C("a.root,b.root,c.root","tree")'
//
// Notes:
// - Assumes each mod*_ph branch has exactly 1 entry per file.
// - "EMPTY" means Length$(modX_ph)==0 at entry 0; "NONEMPTY" means Length$>0.
// - Does NOT split ALL_ZERO vs HAS_NONZERO; this is empty vs non-empty only.

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TObjArray.h"
#include "TString.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLegend.h"

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <cctype>

namespace {
  // -------- small helpers --------
  double IntegralWidth(const TH1* h) {
    if (!h) return 0.0;
    double s = 0.0;
    const int nb = h->GetNbinsX();
    for (int i = 1; i <= nb; ++i) s += h->GetBinContent(i) * h->GetBinWidth(i);
    return s;
  }
  void WidenRange(double& lo, double& hi) {
    if (lo == hi) { lo = std::max(0.0, lo - 1.0); hi = lo + 2.0; }
    const double span = hi - lo;
    lo = std::max(0.0, lo - 0.05 * span);
    hi = hi + 0.05 * span;
  }
  std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
      // trim spaces
      size_t b = 0, e = item.size();
      while (b < e && std::isspace(static_cast<unsigned char>(item[b]))) ++b;
      while (e > b && std::isspace(static_cast<unsigned char>(item[e-1]))) --e;
      if (e > b) out.emplace_back(item.substr(b, e-b));
    }
    return out;
  }
  std::vector<std::string> read_list_file(const std::string& path) {
    std::vector<std::string> out;
    std::ifstream fin(path);
    if (!fin) return out;
    std::string line;
    while (std::getline(fin, line)) {
      // allow comments and blanks
      auto pos = line.find('#');
      if (pos != std::string::npos) line = line.substr(0, pos);
      size_t b = 0, e = line.size();
      while (b < e && std::isspace(static_cast<unsigned char>(line[b]))) ++b;
      while (e > b && std::isspace(static_cast<unsigned char>(line[e-1]))) --e;
      if (e > b) out.emplace_back(line.substr(b, e-b));
    }
    return out;
  }
  void range_from(const std::vector<double>& a, const std::vector<double>& b, double& lo, double& hi) {
    lo = +1e300; hi = -1e300;
    for (double v : a) { if (v < lo) lo = v; if (v > hi) hi = v; }
    for (double v : b) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (lo == +1e300) { lo = 0.0; hi = 1.0; }
    WidenRange(lo, hi);
  }
}

// core worker for one file; appends to accumulators
static void _process_one_file(const std::string& in_file,
                              const char* tree_name,
                              std::vector<double>& tot_empty,
                              std::vector<double>& tot_nonempty,
                              std::vector<double>& zip_empty,
                              std::vector<double>& zip_nonempty,
                              std::vector<std::pair<double,double>>& pairs_all) {
  TFile* fin = TFile::Open(in_file.c_str(), "READ");
  if (!fin || fin->IsZombie()) {
    std::cerr << "[WARN] Cannot open: " << in_file << std::endl;
    if (fin) fin->Close();
    return;
  }
  TTree* t = dynamic_cast<TTree*>(fin->Get(tree_name));
  if (!t) {
    std::cerr << "[WARN] Tree not found in " << in_file << " : " << tree_name << std::endl;
    fin->Close();
    return;
  }
  if (t->GetEntries() <= 0) {
    std::cerr << "[INFO] Tree has no entries: " << in_file << std::endl;
    fin->Close();
    return;
  }

  TObjArray* blist = t->GetListOfBranches();
  const int nbranches = blist ? blist->GetEntriesFast() : 0;

  for (int i = 0; i < nbranches; ++i) {
    TBranch* br = dynamic_cast<TBranch*>(blist->UncheckedAt(i));
    if (!br) continue;
    TString name = br->GetName();
    if (!name.BeginsWith("mod") || !name.EndsWith("_ph")) continue;

    // Bytes stats (no decompression)
    const Long64_t tot = br->GetTotBytes();
    const Long64_t zip = br->GetZipBytes();

    // Classify EMPTY vs NONEMPTY using Length$ at entry 0 (decompress only this branch)
    t->SetBranchStatus("*", 0);
    t->SetBranchStatus(name, 1);
    TString eLen = "Length$(" + name + ")";
    t->Draw(eLen, "", "goff", 1, 0); // one entry, starting at 0
    double L = 0.0;
    if (t->GetSelectedRows() > 0) L = t->GetV1()[0];

    if (L <= 0.0) {
      tot_empty.push_back(static_cast<double>(tot));
      zip_empty.push_back(static_cast<double>(zip));
      pairs_all.emplace_back(static_cast<double>(tot), static_cast<double>(zip));
    } else {
      tot_nonempty.push_back(static_cast<double>(tot));
      zip_nonempty.push_back(static_cast<double>(zip));
      pairs_all.emplace_back(static_cast<double>(tot), static_cast<double>(zip));
    }
  }

  // restore status to avoid surprising later users (optional)
  t->SetBranchStatus("*", 1);
  fin->Close();
}

// Main entry — can accept either a single ROOT file, a comma-separated list, or a text list file.
void ph_branch_bytes(const char* in_files_or_list,
                                   const char* tree_name,
                                   const char* out_file = "merged_bytes.root",
                                   bool save_png = true) {
  // Resolve inputs:
  // - if it's a .txt/.list file → read as list
  // - else if contains ',' → split CSV
  // - else → treat as single file
  std::vector<std::string> inputs;
  std::string s = in_files_or_list ? std::string(in_files_or_list) : std::string();
  if (s.size() >= 4) {
    std::string ext = s.substr(s.find_last_of('.') + 1);
    for (auto& c : ext) c = std::tolower(static_cast<unsigned char>(c));
    if (ext == "txt" || ext == "list") {
      inputs = read_list_file(s);
    }
  }
  if (inputs.empty()) {
    auto v = split_csv(s);
    if (!v.empty()) inputs = std::move(v);
  }
  if (inputs.empty() && !s.empty()) {
    inputs.push_back(s); // single file path
  }

  if (inputs.empty()) {
    std::cerr << "[ERROR] No input files resolved from: " << s << std::endl;
    return;
  }

  // Accumulators across files
  std::vector<double> tot_empty, tot_nonempty, zip_empty, zip_nonempty;
  std::vector<std::pair<double,double>> pairs_all;
  tot_empty.reserve(4096); tot_nonempty.reserve(4096);
  zip_empty.reserve(4096); zip_nonempty.reserve(4096);
  pairs_all.reserve(8192);

  // Process each file
  size_t n_ok = 0;
  for (const auto& f : inputs) {
    _process_one_file(f, tree_name, tot_empty, tot_nonempty, zip_empty, zip_nonempty, pairs_all);
    ++n_ok;
  }
  std::cout << "[INFO] Processed files: " << n_ok
            << "  EMPTY=" << tot_empty.size()
            << "  NONEMPTY=" << tot_nonempty.size() << std::endl;

  if (tot_empty.empty() && tot_nonempty.empty()) {
    std::cerr << "[INFO] No mod*_ph branches found across inputs." << std::endl;
    return;
  }

  // Histogram ranges from combined samples
  double minTot, maxTot, minZip, maxZip;
  range_from(tot_empty, tot_nonempty, minTot, maxTot);
  range_from(zip_empty, zip_nonempty, minZip, maxZip);

  const int nb1 = 200;

  // Create and fill histograms
  TH1D* hTotE = new TH1D("hTotBytes_empty",
                         "mod*_ph TotBytes (EMPTY vs NONEMPTY, merged);TotBytes [bytes];Branches",
                         nb1, minTot, maxTot);
  TH1D* hTotN = new TH1D("hTotBytes_nonempty",
                         "mod*_ph TotBytes (EMPTY vs NONEMPTY, merged);TotBytes [bytes];Branches",
                         nb1, minTot, maxTot);
  TH1D* hZipE = new TH1D("hZipBytes_empty",
                         "mod*_ph ZipBytes (EMPTY vs NONEMPTY, merged);ZipBytes [bytes];Branches",
                         nb1, minZip, maxZip);
  TH1D* hZipN = new TH1D("hZipBytes_nonempty",
                         "mod*_ph ZipBytes (EMPTY vs NONEMPTY, merged);ZipBytes [bytes];Branches",
                         nb1, minZip, maxZip);

  for (double v : tot_empty)    hTotE->Fill(v);
  for (double v : tot_nonempty) hTotN->Fill(v);
  for (double v : zip_empty)    hZipE->Fill(v);
  for (double v : zip_nonempty) hZipN->Fill(v);

  // Density clones
  auto make_density = [](TH1D* h, const char* newname) -> TH1D* {
    TH1D* c = (TH1D*)h->Clone(newname);
    double S = IntegralWidth(c);
    if (S > 0) c->Scale(1.0 / S);
    return c;
  };
  TH1D* hTotE_d = make_density(hTotE, "hTotBytes_empty_density");
  TH1D* hTotN_d = make_density(hTotN, "hTotBytes_nonempty_density");
  TH1D* hZipE_d = make_density(hZipE, "hZipBytes_empty_density");
  TH1D* hZipN_d = make_density(hZipN, "hZipBytes_nonempty_density");

  // Style
  hTotE_d->SetLineColor(kRed+1);  hTotE_d->SetLineWidth(2);
  hTotN_d->SetLineColor(kBlue+1); hTotN_d->SetLineWidth(2);
  hZipE_d->SetLineColor(kRed+1);  hZipE_d->SetLineWidth(2);
  hZipN_d->SetLineColor(kBlue+1); hZipN_d->SetLineWidth(2);

  // Overlays
  gStyle->SetOptStat(0);

  TCanvas c1("cTot","TotBytes overlay (merged)",900,650);
  hTotN_d->SetTitle("mod*_ph TotBytes — density overlay (merged);TotBytes [bytes];Density");
  hTotN_d->Draw("HIST");
  hTotE_d->Draw("HIST SAME");
  {
    TLegend leg(0.65,0.75,0.88,0.90);
    leg.AddEntry(hTotN_d, Form("NONEMPTY (n=%zu)", tot_nonempty.size()), "l");
    leg.AddEntry(hTotE_d, Form("EMPTY (n=%zu)", tot_empty.size()), "l");
    leg.Draw();
  }

  TCanvas c2("cZip","ZipBytes overlay (merged)",900,650);
  hZipN_d->SetTitle("mod*_ph ZipBytes — density overlay (merged);ZipBytes [bytes];Density");
  hZipN_d->Draw("HIST");
  hZipE_d->Draw("HIST SAME");
  {
    TLegend leg(0.65,0.75,0.88,0.90);
    leg.AddEntry(hZipN_d, Form("NONEMPTY (n=%zu)", zip_nonempty.size()), "l");
    leg.AddEntry(hZipE_d, Form("EMPTY (n=%zu)", zip_empty.size()), "l");
    leg.Draw();
  }

  // 2D Zip vs Tot for all branches (merged)
  double loT = minTot, hiT = maxTot, loZ = minZip, hiZ = maxZip;
  TH2D* hZipVsTot = new TH2D("hZipVsTot",
                             "ZipBytes vs TotBytes (mod*_ph, merged);TotBytes [bytes];ZipBytes [bytes]",
                             150, loT, hiT, 150, loZ, hiZ);
  for (const auto& p : pairs_all) hZipVsTot->Fill(p.first, p.second);

  TCanvas c3("cZipTot","Zip vs Tot (merged)",900,700);
  c3.SetRightMargin(0.15);
  hZipVsTot->Draw("COLZ");

  // Save everything
  TFile fout(out_file, "RECREATE");
  hTotE->Write(); hTotN->Write(); hZipE->Write(); hZipN->Write();
  hTotE_d->Write(); hTotN_d->Write(); hZipE_d->Write(); hZipN_d->Write();
  hZipVsTot->Write();
  c1.Write("cTot_overlay_merged");
  c2.Write("cZip_overlay_merged");
  c3.Write("cZipVsTot_merged");
  fout.Close();
  
  std::cout<<"For Empty branches: TotBytes min/mean/max = "
           <<(tot_empty.empty() ? 0.0 : *std::min_element(tot_empty.begin(), tot_empty.end()))<<"/"
           <<(tot_empty.empty() ? 0.0 : std::accumulate(tot_empty.begin(), tot_empty.end(), 0.0)/tot_empty.size())<<"/"
           <<(tot_empty.empty() ? 0.0 : *std::max_element(tot_empty.begin(), tot_empty.end()))<<std::endl;
  std::cout<<"For Empty branches: ZipBytes min/mean/max = "
            <<(zip_empty.empty() ? 0.0 : *std::min_element(zip_empty.begin(), zip_empty.end()))<<"/"
            <<(zip_empty.empty() ? 0.0 : std::accumulate(zip_empty.begin(), zip_empty.end(), 0.0)/zip_empty.size())<<"/"
            <<(zip_empty.empty() ? 0.0 : *std::max_element(zip_empty.begin(), zip_empty.end()))<<std::endl;
  std::cout<<"For NonEmpty branches: TotBytes min/mean/max = "
           <<(tot_nonempty.empty() ? 0.0 : *std::min_element(tot_nonempty.begin(), tot_nonempty.end()))<<"/"
           <<(tot_nonempty.empty() ? 0.0 : std::accumulate(tot_nonempty.begin(), tot_nonempty.end(), 0.0)/tot_nonempty.size())<<"/"
           <<(tot_nonempty.empty() ? 0.0 : *std::max_element(tot_nonempty.begin(), tot_nonempty.end()))<<std::endl;
  std::cout<<"For NonEmpty branches: ZipBytes min/mean/max = "
           <<(zip_nonempty.empty() ? 0.0 : *std::min_element(zip_nonempty.begin(), zip_nonempty.end()))<<"/"
           <<(zip_nonempty.empty() ? 0.0 : std::accumulate(zip_nonempty.begin(), zip_nonempty.end(), 0.0)/zip_nonempty.size())<<"/"
           <<(zip_nonempty.empty() ? 0.0 : *std::max_element(zip_nonempty.begin(), zip_nonempty.end()))<<std::endl;
  std::cout << "[DONE] Wrote merged histograms & canvases to: " << out_file << std::endl;

  if (save_png) {
    c1.SaveAs("hTotBytes_overlay_merged.png");
    c2.SaveAs("hZipBytes_overlay_merged.png");
    c3.SaveAs("hZipVsTot_merged.png");
    std::cout << "[PNG] Saved: hTotBytes_overlay_merged.png, hZipBytes_overlay_merged.png, hZipVsTot_merged.png" << std::endl;
  }
}
