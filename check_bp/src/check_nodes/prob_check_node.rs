use super::*;
use crate::bin_tree::BinMultTree;
use crate::bin_var_node::{CtrlMsg, CtrlMsgA};
use crate::check_msg::CheckMsg;
use crate::check_nodes::CmpOperator;
use belief_propagation::{BPError, BPResult, NodeFunction, NodeIndex};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

pub struct ProbCheckNode<const K: usize, const ETA: usize> {
    n: usize,
    coeffs: [i16; K],
    value: i16,
    op: CmpOperator,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
    connections: Vec<usize>,
    prob_correct: f64,
}

impl<const K: usize, const ETA: usize> ProbCheckNode<K, ETA> {
    pub fn new(
        coeffs: [i16; K],
        value: i16,
        op: CmpOperator,
        n: usize,
        prob_correct: f64,
    ) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);
        Self {
            n,
            coeffs,
            value,
            op,
            connections: Vec::with_capacity(K),
            fft,
            ifft,
            prob_correct
        }
    }
    fn node_function_normal(
        &self,
        inbox: Vec<(NodeIndex, CheckMsg<ETA>)>,
    ) -> BPResult<Vec<(NodeIndex, CheckMsg<ETA>)>> {
        //println!("{:?}", inbox[0].1.data.to_vec());
        let leafs: Vec<Vec<Complex<f64>>> = inbox
            .iter()
            .map(|(node_index, msg)| {
                msg.mult_and_transform(self.coeffs[*node_index], self.n, &self.fft)
                    .iter()
                    .map(|p| p / (self.n as f64).sqrt())
                    .collect()
            })
            .collect();

        let products: Vec<Vec<Complex<f64>>> =
            BinMultTree::new(leafs, multiply_pointwise).calculate();

        let partials: Vec<Vec<f64>> = products
            .into_iter()
            .map(|prd| ifft(prd, &self.ifft))
            .collect();

        //println!("{:?}", partials);
        let res: Vec<(NodeIndex, CheckMsg<ETA>)> = match self.op {
            CmpOperator::GreaterEq => partials
                .into_iter()
                .zip(inbox.into_iter())
                .map(|(dist_sum, ib)| {
                    (
                        ib.0,
                        derive_from_inequality_greater_prob(
                            dist_sum,
                            self.value,
                            self.coeffs[ib.0],
                            self.prob_correct,
                        ),
                    )
                })
                .collect(),

            CmpOperator::SmallerEq => partials
                .into_iter()
                .zip(inbox.into_iter())
                .map(|(dist_sum, ib)| {
                    (
                        ib.0,
                        derive_from_inequality_smaller_prob(
                            dist_sum,
                            self.value,
                            self.coeffs[ib.0],
                            self.prob_correct,
                        ),
                    )
                })
                .collect(),
        };
        Ok(res)
    }
}

impl<const K: usize, const ETA: usize> NodeFunction<i16, CheckMsg<ETA>, CtrlMsg, CtrlMsgA>
    for ProbCheckNode<K, ETA>
{
    fn node_function(
        &mut self,
        inbox: Vec<(NodeIndex, CheckMsg<ETA>)>,
    ) -> BPResult<Vec<(NodeIndex, CheckMsg<ETA>)>> {
        self.node_function_normal(inbox)
    }

    fn number_inputs(&self) -> Option<usize> {
        Some(K)
    }

    fn is_factor(&self) -> bool {
        true
    }
    fn get_prior(&self) -> Option<CheckMsg<ETA>> {
        None
    }
    fn initialize(&mut self, connections: Vec<NodeIndex>) -> BPResult<()> {
        //TODO: Ensure connections are sorted
        if connections.len() != K {
            Err(BPError::new(
                "ProbCheckNode::initialize".to_owned(),
                format!(
                    "Wrong number ({}) of connections given ({}).",
                    K,
                    connections.len()
                ),
            ))
        } else {
            self.connections = connections;
            Ok(())
        }
    }
    fn reset(&mut self) -> BPResult<()> {
        Ok(())
    }

    fn is_ready(
        &self,
        recv_from: &Vec<(NodeIndex, CheckMsg<ETA>)>,
        _current_step: usize,
    ) -> BPResult<bool> {
        Ok(recv_from.len() == self.connections.len())
    }
}
