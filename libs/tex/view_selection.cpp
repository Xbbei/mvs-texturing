/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <util/timer.h>

#include "util.h"
#include "texturing.h"
#include "mapmap/full.h"

TEX_NAMESPACE_BEGIN

void
view_selection(DataCosts const & data_costs, UniGraph * graph, const std::vector<std::map<uint32_t, math::Vec4f> >& faces_colors, Settings const &) {
    using uint_t = unsigned int;
    using cost_t = float;
    constexpr uint_t simd_w = mapmap::sys_max_simd_width<cost_t>();
    using unary_t = mapmap::UnaryTable<cost_t, simd_w>;
    using pairwise_t = mapmap::PairwisePotts<cost_t, simd_w>;

    /* Construct graph */
    mapmap::Graph<cost_t> mgraph(graph->num_nodes());

    for (std::size_t i = 0; i < graph->num_nodes(); ++i) {
        if (data_costs.col(i).empty()) continue;

        std::vector<std::size_t> adj_faces = graph->get_adj_nodes(i);
        for (std::size_t j = 0; j < adj_faces.size(); ++j) {
            std::size_t adj_face = adj_faces[j];
            if (data_costs.col(adj_face).empty()) continue;

            /* Uni directional */
            if (i < adj_face) {
                mgraph.add_edge(i, adj_face, 1.0f);
            }
        }
    }
    mgraph.update_components();

    mapmap::LabelSet<cost_t, simd_w> label_set(graph->num_nodes(), false);
    for (std::size_t i = 0; i < data_costs.cols(); ++i) {
        DataCosts::Column const & data_costs_for_node = data_costs.col(i);

        std::vector<mapmap::_iv_st<cost_t, simd_w> > labels;
        if (data_costs_for_node.empty()) {
            labels.push_back(0);
        } else {
            labels.resize(data_costs_for_node.size());
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j) {
                labels[j] = data_costs_for_node[j].first + 1;
            }
        }

        label_set.set_label_set_for_node(i, labels);
    }

    std::vector<unary_t> unaries;
    unaries.reserve(data_costs.cols());
    pairwise_t pairwise(1.0f);
    for (std::size_t i = 0; i < data_costs.cols(); ++i) {
        DataCosts::Column const & data_costs_for_node = data_costs.col(i);

        std::vector<mapmap::_s_t<cost_t, simd_w> > costs;
        if (data_costs_for_node.empty()) {
            costs.push_back(1.0f);
        } else {
            costs.resize(data_costs_for_node.size());
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j) {
                float cost = data_costs_for_node[j].second;
                costs[j] = cost;
            }

        }

        unaries.emplace_back(i, &label_set);
        unaries.back().set_costs(costs);
    }

    mapmap::StopWhenReturnsDiminish<cost_t, simd_w> terminate(5, 0.01);
    std::vector<mapmap::_iv_st<cost_t, simd_w> > solution;

    auto display = [](const mapmap::luint_t time_ms,
            const mapmap::_iv_st<cost_t, simd_w> objective) {
        std::cout << "\t\t" << time_ms / 1000 << "\t" << objective << std::endl;
    };

    /* Create mapMAP solver object. */
    mapmap::mapMAP<cost_t, simd_w> solver;
    solver.set_graph(&mgraph);
    solver.set_label_set(&label_set);
    for(std::size_t i = 0; i < graph->num_nodes(); ++i)
        solver.set_unary(i, &unaries[i]);
    solver.set_pairwise(&pairwise);
    solver.set_logging_callback(display);
    solver.set_termination_criterion(&terminate);

    /* Pass configuration arguments (optional) for solve. */
    mapmap::mapMAP_control ctr;
    ctr.use_multilevel = true;
    ctr.use_spanning_tree = true;
    ctr.use_acyclic = true;
    ctr.spanning_tree_multilevel_after_n_iterations = 5;
    ctr.force_acyclic = true;
    ctr.min_acyclic_iterations = 5;
    ctr.relax_acyclic_maximal = true;
    ctr.tree_algorithm = mapmap::LOCK_FREE_TREE_SAMPLER;

    /* Set false for non-deterministic (but faster) mapMAP execution. */
    ctr.sample_deterministic = true;
    ctr.initial_seed = 548923723;

    std::cout << "\tOptimizing:\n\t\tTime[s]\tEnergy" << std::endl;
    solver.optimize(solution, ctr);

    /* Label 0 is undefined. */
    std::size_t num_labels = data_costs.rows() + 1;
    std::size_t undefined = 0;
    /* Extract resulting labeling from solver. */
    for (std::size_t i = 0; i < graph->num_nodes(); ++i) {
        int label = label_set.label_from_offset(i, solution[i]);
        if (label < 0 || num_labels <= static_cast<std::size_t>(label)) {
            throw std::runtime_error("Incorrect labeling");
        }
        if (label == 0) undefined += 1;
        graph->set_label(i, static_cast<std::size_t>(label));
    }
    std::cout << '\t' << undefined << " faces have not been seen" << std::endl;
}


void
view_selection_table(DataCosts const & data_costs, UniGraph * graph, const std::vector<std::map<uint32_t, math::Vec4f> >& faces_colors, Settings const &) {
    using uint_t = unsigned int;
    using cost_t = float;
    constexpr uint_t simd_w = mapmap::sys_max_simd_width<cost_t>();
    using unary_t = mapmap::UnaryTable<cost_t, simd_w>;
    using pairwise_t = mapmap::PairwiseTable<cost_t, simd_w>;

    /* Construct graph */
    mapmap::Graph<cost_t> mgraph(graph->num_nodes());
    for (std::size_t i = 0; i < graph->num_nodes(); ++i) {
        if (data_costs.col(i).empty()) continue;

        std::vector<std::size_t> adj_faces = graph->get_adj_nodes(i);
        for (std::size_t j = 0; j < adj_faces.size(); ++j) {
            std::size_t adj_face = adj_faces[j];
            if (data_costs.col(adj_face).empty()) continue;

            /* Uni directional */
            if (i < adj_face) {
                mgraph.add_edge(i, adj_face, 1.0f);
            }
        }
    }
    mgraph.update_components();

    mapmap::LabelSet<cost_t, simd_w> label_set(graph->num_nodes(), false);
    for (std::size_t i = 0; i < data_costs.cols(); ++i) {
        DataCosts::Column const & data_costs_for_node = data_costs.col(i);

        std::vector<mapmap::_iv_st<cost_t, simd_w> > labels;
        if (data_costs_for_node.empty()) {
            labels.push_back(0);
        } else {
            labels.resize(data_costs_for_node.size());
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j) {
                labels[j] = data_costs_for_node[j].first + 1;
            }
        }

        label_set.set_label_set_for_node(i, labels);
    }

    std::vector<unary_t> unaries;
    unaries.reserve(data_costs.cols());
    for (std::size_t i = 0; i < data_costs.cols(); ++i) {
        DataCosts::Column const & data_costs_for_node = data_costs.col(i);

        std::vector<mapmap::_s_t<cost_t, simd_w> > costs;
        if (data_costs_for_node.empty()) {
            costs.push_back(1.0f);
        } else {
            costs.resize(data_costs_for_node.size());
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j) {
                float cost = data_costs_for_node[j].second;
                costs[j] = cost;
            }

        }

        unaries.emplace_back(i, &label_set);
        unaries.back().set_costs(costs);
    }

    mapmap::StopWhenReturnsDiminish<cost_t, simd_w> terminate(10, 0.001);
    std::vector<mapmap::_iv_st<cost_t, simd_w> > solution;

    auto display = [](const mapmap::luint_t time_ms,
            const mapmap::_iv_st<cost_t, simd_w> objective) {
        std::cout << "\t\t" << time_ms / 1000 << "\t" << objective << std::endl;
    };

    /* Create mapMAP solver object. */
    mapmap::mapMAP<cost_t, simd_w> solver;
    solver.set_graph(&mgraph);
    solver.set_label_set(&label_set);
    for(std::size_t i = 0; i < graph->num_nodes(); ++i)
        solver.set_unary(i, &unaries[i]);
    // solver.set_pairwise(&pairwise);
    std::vector<mapmap::GraphEdge<cost_t> > medges = mgraph.edges();
    std::vector<std::shared_ptr<pairwise_t> > pairs;
    for (std::size_t i = 0; i < mgraph.num_edges(); ++i) {
        mapmap::GraphEdge<cost_t> edge = medges[i];
        uint32_t nodea = edge.node_a;
        uint32_t nodeb = edge.node_b;
        std::vector<mapmap::_s_t<cost_t, simd_w> > table;
        DataCosts::Column const & data_costs_for_nodei = data_costs.col(nodea);
        DataCosts::Column const & data_costs_for_nodej = data_costs.col(nodeb);
        table.resize(data_costs_for_nodei.size() * data_costs_for_nodej.size());
        for (std::size_t m = 0; m < data_costs_for_nodei.size(); ++m) {
            uint16_t labeli = data_costs_for_nodei[m].first;
            for (std::size_t n = 0; n < data_costs_for_nodej.size(); ++n) {
                uint16_t labelj = data_costs_for_nodej[n].first;
                if (labeli == labelj) {
                    table[m * data_costs_for_nodej.size() + n] = 0;
                    continue;
                }
                math::Vec4f icolor = faces_colors[nodea].at(labeli);
                math::Vec4f jcolor = faces_colors[nodeb].at(labelj);
                math::Vec3f diffcolor = math::Vec3f(icolor[0] - jcolor[0], icolor[1] - jcolor[1], icolor[2] - jcolor[2]);
                // cost_t diff = (diffcolor.norm() + icolor[3] + jcolor[3]) * 10.0;
                cost_t diff = diffcolor.norm() * 10;
                // diff = diff * diff;
                if (diff > 10.0)
                    diff = 10.0;
                table[m * data_costs_for_nodej.size() + n] = diff;
            }
        }
        std::shared_ptr<pairwise_t> pairwise;
        pairwise = std::shared_ptr<pairwise_t>(new pairwise_t(nodea, nodeb, &label_set, table));
        pairs.push_back(pairwise);
        solver.set_pairwise(i, pairs[i].get());
    }
    solver.set_logging_callback(display);
    solver.set_termination_criterion(&terminate);

    /* Pass configuration arguments (optional) for solve. */
    mapmap::mapMAP_control ctr;
    ctr.use_multilevel = true;
    ctr.use_spanning_tree = true;
    ctr.use_acyclic = true;
    ctr.spanning_tree_multilevel_after_n_iterations = 5;
    ctr.force_acyclic = true;
    ctr.min_acyclic_iterations = 5;
    ctr.relax_acyclic_maximal = true;
    ctr.tree_algorithm = mapmap::LOCK_FREE_TREE_SAMPLER;

    /* Set false for non-deterministic (but faster) mapMAP execution. */
    ctr.sample_deterministic = true;
    ctr.initial_seed = 548923723;

    std::cout << "\tOptimizing:\n\t\tTime[s]\tEnergy" << std::endl;
    solver.optimize(solution, ctr);

    /* Label 0 is undefined. */
    std::size_t num_labels = data_costs.rows() + 1;
    std::size_t undefined = 0;
    /* Extract resulting labeling from solver. */
    for (std::size_t i = 0; i < graph->num_nodes(); ++i) {
        int label = label_set.label_from_offset(i, solution[i]);
        if (label < 0 || num_labels <= static_cast<std::size_t>(label)) {
            throw std::runtime_error("Incorrect labeling");
        }
        if (label == 0) undefined += 1;
        graph->set_label(i, static_cast<std::size_t>(label));
    }
    std::cout << '\t' << undefined << " faces have not been seen" << std::endl;
}

TEX_NAMESPACE_END
